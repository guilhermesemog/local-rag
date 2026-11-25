const askButton = document.getElementById('askButton');
const restartButton = document.getElementById('restartButton');
const uploadFileButton = document.getElementById('uploadFileButton');
const clearConversationButton = document.getElementById('clearConversationButton');

const fileInput = document.getElementById('fileInput');
const replaceFileInput = document.getElementById('replaceFileInput');
const questionInput = document.getElementById('questionInput');

const documentsList = document.getElementById('documentsList');
const documentSummary = document.getElementById('documentSummary');
const filterSummary = document.getElementById('filterSummary');
const conversationList = document.getElementById('conversationList');
const emptyConversation = document.getElementById('emptyConversation');
const sessionLabel = document.getElementById('sessionLabel');

const uploadLoading = document.getElementById('uploadLoading');
const restartLoading = document.getElementById('restartLoading');

const emptyMessage = document.getElementById('emptyMessage');
const appStatus = document.getElementById('appStatus');
const adminKeyDialog = document.getElementById('adminKeyDialog');
const adminKeyInput = document.getElementById('adminKeyInput');

const baseURL = window.location.origin;
const CONVERSATION_KEY = 'localrag.conversation.v1';
const FILTER_KEY = 'localrag.document_filter.v1';
const MAX_STORED_EXCHANGES = 50;
const MAX_CONTEXT_TURNS = 4;
const MAX_CONTEXT_TEXT = 2000;
const MAX_CONTEXT_TOTAL = 6000;
const PHASE_LABELS = {
    checking_index: 'Checking the document index…',
    searching_documents: 'Retrieving relevant context…',
    generating_answer: 'Generating the answer…',
    formatting_result: 'Finishing the answer…',
    completed: 'Answer ready.',
};

let sessionId = '';
let exchanges = [];
let documents = [];
let selectedDocumentIds = new Set();
let databaseIsEmpty = true;
let activeController = null;
let isStreaming = false;
let documentActionPending = false;
let pendingReplaceId = null;
let liveExchange = null;

function setStatus(message, type = '') {
    appStatus.textContent = message;
    appStatus.className = `status${type ? ` ${type}` : ''}`;
}

function setLoading(element, isLoading) {
    element.classList.toggle('hidden', !isLoading);
}

function safeStorageGet(key) {
    try {
        return window.localStorage.getItem(key);
    } catch {
        return null;
    }
}

function safeStorageSet(key, value) {
    try {
        window.localStorage.setItem(key, value);
        return true;
    } catch {
        return false;
    }
}

function createSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID();
    }
    if (window.crypto && typeof window.crypto.getRandomValues === 'function') {
        const bytes = window.crypto.getRandomValues(new Uint8Array(16));
        return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
    }
    return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function saveConversation() {
    safeStorageSet(
        CONVERSATION_KEY,
        JSON.stringify({
            version: 1,
            session_id: sessionId,
            exchanges,
        })
    );
}

function loadConversation() {
    const raw = safeStorageGet(CONVERSATION_KEY);
    if (!raw) {
        sessionId = createSessionId();
        exchanges = [];
        saveConversation();
        return;
    }

    try {
        const data = JSON.parse(raw);
        if (!data || data.version !== 1 || !Array.isArray(data.exchanges)) {
            throw new Error('Invalid conversation data');
        }
        sessionId = typeof data.session_id === 'string' && data.session_id
            ? data.session_id
            : createSessionId();
        exchanges = data.exchanges
            .filter((exchange) => (
                exchange
                && typeof exchange.question === 'string'
                && typeof exchange.answer === 'string'
            ))
            .map((exchange) => ({
                question: exchange.question,
                answer: exchange.answer,
                sources: Array.isArray(exchange.sources) ? exchange.sources : [],
                created_at: typeof exchange.created_at === 'string'
                    ? exchange.created_at
                    : '',
            }))
            .slice(-MAX_STORED_EXCHANGES);
    } catch {
        sessionId = createSessionId();
        exchanges = [];
    }
    saveConversation();
}

function loadFilters() {
    selectedDocumentIds = new Set();
    const raw = safeStorageGet(FILTER_KEY);
    if (!raw) return;
    try {
        const ids = JSON.parse(raw);
        if (Array.isArray(ids)) {
            selectedDocumentIds = new Set(
                ids.filter((documentId) => typeof documentId === 'string')
            );
        }
    } catch {
        selectedDocumentIds = new Set();
    }
}

function saveFilters() {
    safeStorageSet(FILTER_KEY, JSON.stringify([...selectedDocumentIds]));
}

function buildHistory() {
    const selected = [];
    let total = 0;

    for (const exchange of [...exchanges].reverse()) {
        if (typeof exchange.question !== 'string'
            || typeof exchange.answer !== 'string') {
            continue;
        }
        const question = exchange.question.trim().slice(0, MAX_CONTEXT_TEXT);
        const answer = exchange.answer.trim().slice(0, MAX_CONTEXT_TEXT);
        if (!question || !answer) continue;
        const size = question.length + answer.length;
        if (total + size > MAX_CONTEXT_TOTAL) break;
        selected.unshift({ question, answer });
        total += size;
        if (selected.length >= MAX_CONTEXT_TURNS) break;
    }

    return selected;
}

function updateDocumentControls() {
    const disabled = isStreaming || documentActionPending;
    for (const input of documentsList.querySelectorAll('input[type="checkbox"]')) {
        input.disabled = disabled;
    }
    for (const button of documentsList.querySelectorAll('button[data-action]')) {
        button.disabled = disabled;
    }
}

function updateControls() {
    const hasQuestion = questionInput.value.trim() !== '';
    const busy = isStreaming || documentActionPending;
    restartButton.disabled = databaseIsEmpty || busy;
    askButton.disabled = databaseIsEmpty || busy || !hasQuestion;
    clearConversationButton.disabled = busy || exchanges.length === 0;
    updateDocumentControls();
}

async function readJson(response) {
    try {
        return await response.json();
    } catch {
        return {};
    }
}

function errorMessage(data, fallback) {
    const detail = data && data.detail;
    if (typeof detail === 'string' && detail) {
        return detail;
    }
    if (Array.isArray(detail)) {
        const messages = detail
            .map((entry) => (
                entry && typeof entry.msg === 'string' ? entry.msg : null
            ))
            .filter(Boolean);
        if (messages.length) {
            return messages.join(' ');
        }
    }
    return fallback;
}

function updateFilterSummary() {
    const selected = documents.filter(
        (document) => document.document_id
            && selectedDocumentIds.has(document.document_id)
    );
    if (!documents.length) {
        filterSummary.textContent = 'No documents available.';
    } else if (!selected.length) {
        filterSummary.textContent = 'Searching all documents.';
    } else {
        const names = selected
            .map((document) => document.document_name || 'Unknown document')
            .join(', ');
        filterSummary.textContent = `Searching ${selected.length} document${selected.length === 1 ? '' : 's'}: ${names}`;
    }
}

function renderSourcesInto(list, sources) {
    list.replaceChildren();
    const ordered = [...(sources || [])].sort((left, right) => {
        const pageDifference = (left.page ?? 0) - (right.page ?? 0);
        return pageDifference || (left.chunk ?? 0) - (right.chunk ?? 0);
    });

    for (const source of ordered) {
        const item = document.createElement('li');
        const label = document.createElement('strong');
        label.textContent = source.document_name || 'Unknown document';
        item.append(label);

        const details = [];
        if (source.page != null) details.push(`Page ${source.page}`);
        if (source.chunk != null) details.push(`Chunk ${source.chunk}`);
        if (typeof source.similarity === 'number') {
            details.push(`${Math.round(source.similarity * 100)}% match`);
        }
        if (details.length) {
            const meta = document.createElement('span');
            meta.className = 'sourceMeta';
            meta.textContent = details.join(' · ');
            item.append(meta);
        }

        if (source.text) {
            const excerpt = document.createElement('blockquote');
            excerpt.textContent = source.text;
            item.append(excerpt);
        }

        list.append(item);
    }
}

function createExchangeElement(exchange, { live = false } = {}) {
    const item = document.createElement('li');
    item.className = `exchange${live ? ' live' : ''}`;

    const question = document.createElement('p');
    question.className = 'exchangeQuestion';
    const role = document.createElement('span');
    role.className = 'exchangeRole';
    role.textContent = 'You';
    question.append(role, document.createTextNode(exchange.question));
    item.append(question);

    let progressRow = null;
    let progressBar = null;
    let phase = null;
    if (live) {
        progressRow = document.createElement('div');
        progressRow.className = 'progressRow';
        progressBar = document.createElement('progress');
        progressBar.max = 100;
        progressBar.value = 0;
        progressBar.setAttribute('aria-label', 'Answer progress');
        phase = document.createElement('p');
        phase.className = 'phase';
        phase.setAttribute('role', 'status');
        phase.setAttribute('aria-live', 'polite');
        progressRow.append(progressBar, phase);
        item.append(progressRow);
        item.setAttribute('aria-busy', 'true');
    }

    const answer = document.createElement('div');
    answer.className = 'exchangeAnswer';
    answer.textContent = exchange.answer || '';
    item.append(answer);

    const copyRow = document.createElement('div');
    copyRow.className = 'copyRow';
    copyRow.classList.toggle('hidden', !(exchange.answer || '').trim());
    const copyButton = document.createElement('button');
    copyButton.type = 'button';
    copyButton.className = 'copyButton';
    copyButton.textContent = 'Copy answer';
    copyButton.addEventListener('click', () => {
        void copyAnswerText(exchange.answer, copyButton);
    });
    copyRow.append(copyButton);
    item.append(copyRow);

    const sourcesSection = document.createElement('section');
    sourcesSection.className = 'sources';
    const sourcesHeading = document.createElement('h4');
    sourcesHeading.textContent = 'Sources';
    const sourcesList = document.createElement('ol');
    sourcesSection.append(sourcesHeading, sourcesList);
    sourcesSection.classList.toggle('hidden', !(exchange.sources || []).length);
    renderSourcesInto(sourcesList, exchange.sources || []);
    item.append(sourcesSection);

    const error = document.createElement('p');
    error.className = 'errorMessage hidden';
    error.setAttribute('role', 'alert');
    item.append(error);

    return {
        item,
        exchange,
        answer,
        copyRow,
        copyButton,
        sourcesSection,
        sourcesList,
        progressRow,
        progressBar,
        phase,
        error,
    };
}

function renderConversation() {
    conversationList.replaceChildren();
    for (const exchange of exchanges) {
        conversationList.append(createExchangeElement(exchange).item);
    }
    emptyConversation.classList.toggle('hidden', exchanges.length > 0);
    sessionLabel.textContent = sessionId
        ? `Session ${sessionId.slice(0, 8)}… · History is saved in this browser.`
        : '';
    updateControls();
}

function scrollToLatest(element) {
    if (typeof element.scrollIntoView !== 'function') return;
    const reduceMotion = window.matchMedia
        && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    element.scrollIntoView({
        block: 'nearest',
        behavior: reduceMotion ? 'auto' : 'smooth',
    });
}

function startLiveExchange(question) {
    const exchange = {
        question,
        answer: '',
        sources: [],
        created_at: new Date().toISOString(),
    };
    const refs = createExchangeElement(exchange, { live: true });
    conversationList.append(refs.item);
    emptyConversation.classList.add('hidden');
    liveExchange = refs;
    liveExchange.text = '';
    scrollToLatest(refs.item);
}

function showProgress() {
    if (!liveExchange) return;
    liveExchange.progressRow.classList.remove('hidden');
    liveExchange.item.setAttribute('aria-busy', 'true');
}

function setProgress(phase, progress) {
    if (!liveExchange) return;
    showProgress();
    const value = Number(progress);
    liveExchange.progressBar.value = Number.isFinite(value)
        ? Math.max(0, Math.min(100, value))
        : 0;
    liveExchange.phase.textContent = PHASE_LABELS[phase] || 'Processing…';
}

function appendToken(text) {
    if (!liveExchange || !text) return;
    liveExchange.text += text;
    liveExchange.answer.append(document.createTextNode(text));
    liveExchange.copyRow.classList.remove('hidden');
}

function finishAnswer(results, sources) {
    if (!liveExchange) return;
    const text = String(results ?? '').trim();
    const resolvedSources = Array.isArray(sources) ? sources : [];

    liveExchange.exchange.answer = text;
    liveExchange.exchange.sources = resolvedSources;
    liveExchange.answer.textContent = text || 'No answer was returned.';
    liveExchange.copyRow.classList.toggle('hidden', !text);
    liveExchange.progressBar.value = 100;
    liveExchange.phase.textContent = 'Answer ready.';
    liveExchange.item.setAttribute('aria-busy', 'false');
    renderSourcesInto(liveExchange.sourcesList, resolvedSources);
    liveExchange.sourcesSection
        .classList.toggle('hidden', resolvedSources.length === 0);

    exchanges.push(liveExchange.exchange);
    if (exchanges.length > MAX_STORED_EXCHANGES) {
        exchanges = exchanges.slice(-MAX_STORED_EXCHANGES);
    }
    saveConversation();
    liveExchange = null;
    questionInput.value = '';
    setStatus('');
    updateControls();
}

function showError(message) {
    if (!liveExchange) {
        setStatus(message, 'error');
        return;
    }
    liveExchange.error.textContent = message;
    liveExchange.error.classList.remove('hidden');
    liveExchange.item.setAttribute('aria-busy', 'false');
    if (liveExchange.phase) {
        liveExchange.phase.textContent = 'Answer failed.';
    }
}

async function copyAnswerText(text, button) {
    if (!text) return;

    let copied = false;
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            copied = true;
        }
    } catch {
        copied = false;
    }

    if (!copied) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.append(textarea);
        textarea.select();
        try {
            copied = typeof document.execCommand === 'function'
                && document.execCommand('copy');
        } catch {
            copied = false;
        } finally {
            textarea.remove();
        }
    }

    if (!copied) {
        setStatus('The answer could not be copied.', 'error');
        return;
    }

    const previousLabel = button.textContent;
    button.textContent = 'Copied!';
    window.setTimeout(() => {
        button.textContent = previousLabel;
    }, 2000);
}

function parseSseFrame(frame) {
    let event = 'message';
    const dataLines = [];

    for (const line of frame.split('\n')) {
        if (!line || line.startsWith(':')) continue;
        const separator = line.indexOf(':');
        const field = separator === -1 ? line : line.slice(0, separator);
        let value = separator === -1 ? '' : line.slice(separator + 1);
        if (value.startsWith(' ')) value = value.slice(1);
        if (field === 'event') {
            event = value;
        } else if (field === 'data') {
            dataLines.push(value);
        }
    }

    if (!dataLines.length) return null;

    try {
        return { event, data: JSON.parse(dataLines.join('\n')) };
    } catch {
        return null;
    }
}

function requestAdminKey() {
    return new Promise((resolve) => {
        adminKeyInput.value = '';
        adminKeyDialog.returnValue = '';
        adminKeyDialog.addEventListener('close', () => {
            const key = adminKeyDialog.returnValue === 'confirm'
                ? adminKeyInput.value
                : null;
            adminKeyInput.value = '';
            resolve(key);
        }, { once: true });
        adminKeyDialog.showModal();
    });
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForJob(jobId, onUpdate) {
    while (true) {
        const response = await fetch(`${baseURL}/jobs/${encodeURIComponent(jobId)}`);
        const job = await readJson(response);
        if (!response.ok) {
            throw new Error(errorMessage(job, 'Failed to check processing status'));
        }
        onUpdate(job);
        if (job.status === 'failed' || job.status === 'cancelled') {
            throw new Error(job.error || 'Processing task was cancelled.');
        }
        if (job.status === 'completed') {
            return job;
        }
        await delay(1000);
    }
}

async function streamViaPolling(jobId) {
    const completed = await waitForJob(jobId, (job) => {
        setProgress(job.phase, job.progress);
        if (liveExchange && job.result && Array.isArray(job.result.sources)) {
            renderSourcesInto(liveExchange.sourcesList, job.result.sources);
            liveExchange.sourcesSection
                .classList.toggle('hidden', job.result.sources.length === 0);
        }
    });
    const result = typeof completed.result === 'string'
        ? { results: completed.result, sources: [] }
        : (completed.result || {});
    finishAnswer(result.results, result.sources);
}

async function streamQuery(question) {
    const controller = new AbortController();
    activeController = controller;
    isStreaming = true;
    updateControls();
    startLiveExchange(question);
    setProgress('checking_index', 0);

    const payload = { question };
    const selected = [...selectedDocumentIds];
    if (selected.length) payload.document_ids = selected;
    const history = buildHistory();
    if (history.length) payload.history = history;

    let reader = null;
    let jobId = null;
    let sawDone = false;

    try {
        const response = await fetch(`${baseURL}/query/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal,
        });

        if (!response.ok) {
            const data = await readJson(response);
            throw new Error(errorMessage(data, 'Question failed'));
        }

        if (!response.body) {
            const fallback = await fetch(`${baseURL}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal,
            });
            const data = await readJson(fallback);
            if (!fallback.ok) {
                throw new Error(errorMessage(data, 'Question failed'));
            }
            await streamViaPolling(data.id);
            return;
        }

        reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
            let boundary;
            while ((boundary = buffer.indexOf('\n\n')) !== -1) {
                const frame = parseSseFrame(buffer.slice(0, boundary));
                buffer = buffer.slice(boundary + 2);
                if (!frame) continue;

                const { event, data } = frame;
                if (event === 'job') {
                    jobId = data.id;
                } else if (event === 'phase') {
                    setProgress(data.phase, data.progress);
                } else if (event === 'sources') {
                    if (liveExchange) {
                        renderSourcesInto(liveExchange.sourcesList, data.sources || []);
                        liveExchange.sourcesSection
                            .classList.toggle('hidden', !(data.sources || []).length);
                    }
                } else if (event === 'token') {
                    appendToken(data.text || '');
                } else if (event === 'done') {
                    sawDone = true;
                    finishAnswer(data.results, data.sources);
                } else if (event === 'error') {
                    throw new Error(errorMessage(data, 'The answer could not be generated.'));
                }
            }
        }

        if (!sawDone) {
            if (jobId) {
                await streamViaPolling(jobId);
            } else {
                throw new Error('The answer stream ended unexpectedly.');
            }
        }
    } catch (error) {
        if (error.name === 'AbortError') return;
        showError(error.message || 'Question failed');
    } finally {
        if (reader) {
            try {
                await reader.cancel();
            } catch {
                // The stream is already closed.
            }
        }
        liveExchange = null;
        isStreaming = false;
        activeController = null;
        updateControls();
    }
}

async function queryRequest() {
    const question = questionInput.value.trim();

    if (!question) {
        setStatus('Enter a question before submitting.', 'error');
        return;
    }

    if (databaseIsEmpty) {
        setStatus('Upload a document before asking a question.', 'error');
        return;
    }

    if (isStreaming || documentActionPending) return;

    setStatus('');
    await streamQuery(question);
}

function renderDocuments() {
    documentsList.replaceChildren();

    for (const documentEntry of documents) {
        const item = document.createElement('li');
        item.className = 'documentItem';

        if (documentEntry.document_id) {
            const label = document.createElement('label');
            label.className = 'documentCheck';
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = documentEntry.document_id;
            checkbox.dataset.documentId = documentEntry.document_id;
            checkbox.checked = selectedDocumentIds.has(documentEntry.document_id);
            const name = document.createElement('span');
            name.className = 'documentName';
            name.textContent = documentEntry.document_name || 'Unknown document';
            label.append(checkbox, name);
            item.append(label);
        } else {
            const name = document.createElement('span');
            name.className = 'documentName';
            name.textContent = documentEntry.document_name || 'Unknown document';
            item.append(name);
        }

        const meta = document.createElement('span');
        meta.className = 'documentMeta';
        const chunks = Number(documentEntry.chunks) || 0;
        const pages = Number(documentEntry.pages) || 0;
        meta.textContent = `${chunks} chunk${chunks === 1 ? '' : 's'} · ${pages} page${pages === 1 ? '' : 's'}`;
        item.append(meta);

        if (documentEntry.document_id) {
            const actions = document.createElement('div');
            actions.className = 'documentActions';

            const replaceButton = document.createElement('button');
            replaceButton.type = 'button';
            replaceButton.textContent = 'Replace';
            replaceButton.dataset.action = 'replace';
            replaceButton.dataset.documentId = documentEntry.document_id;

            const removeButton = document.createElement('button');
            removeButton.type = 'button';
            removeButton.textContent = 'Remove';
            removeButton.className = 'dangerButton';
            removeButton.dataset.action = 'remove';
            removeButton.dataset.documentId = documentEntry.document_id;

            actions.append(replaceButton, removeButton);
            item.append(actions);
        }

        documentsList.append(item);
    }

    updateDocumentControls();
}

async function refreshDocuments() {
    try {
        const response = await fetch(`${baseURL}/documents`);
        const data = await readJson(response);

        if (!response.ok) {
            throw new Error(errorMessage(data, 'Failed to load documents'));
        }

        documents = Array.isArray(data.documents) ? data.documents : [];
        const chunkCount = Number(data.chunk_count) || 0;
        const documentCount = Number(data.document_count) || 0;
        databaseIsEmpty = chunkCount === 0;

        const validIds = new Set(
            documents
                .map((documentEntry) => documentEntry.document_id)
                .filter(Boolean)
        );
        for (const documentId of [...selectedDocumentIds]) {
            if (!validIds.has(documentId)) {
                selectedDocumentIds.delete(documentId);
            }
        }
        saveFilters();

        documentSummary.textContent = databaseIsEmpty
            ? 'No documents loaded.'
            : `${documentCount} document${documentCount === 1 ? '' : 's'} · ${chunkCount} chunk${chunkCount === 1 ? '' : 's'} loaded.`;
        emptyMessage.textContent = databaseIsEmpty
            ? 'No documents uploaded yet.'
            : 'Documents are ready for questions.';

        renderDocuments();
        updateFilterSummary();
        updateControls();
    } catch (error) {
        documents = [];
        databaseIsEmpty = true;
        renderDocuments();
        updateFilterSummary();
        updateControls();
        setStatus(error.message, 'error');
    }
}

function isPdf(file) {
    return file.type === 'application/pdf'
        || file.name.toLowerCase().endsWith('.pdf');
}

async function removeDocument(documentEntry) {
    if (documentActionPending || isStreaming) return;
    if (!window.confirm(`Remove "${documentEntry.document_name}"? This cannot be undone.`)) {
        return;
    }

    const adminKey = await requestAdminKey();
    if (!adminKey) return;

    documentActionPending = true;
    updateControls();
    try {
        const response = await fetch(
            `${baseURL}/documents/${encodeURIComponent(documentEntry.document_id)}`,
            {
                method: 'DELETE',
                headers: { 'X-Admin-Key': adminKey },
            }
        );
        const data = await readJson(response);
        if (!response.ok) {
            throw new Error(errorMessage(data, 'Document removal failed'));
        }
        setStatus(data.status || 'Document removed.', 'success');
        await refreshDocuments();
    } catch (error) {
        setStatus(error.message, 'error');
    } finally {
        documentActionPending = false;
        updateControls();
    }
}

function startReplace(documentEntry) {
    if (documentActionPending || isStreaming) return;
    pendingReplaceId = documentEntry.document_id;
    replaceFileInput.value = '';
    replaceFileInput.click();
}

async function handleReplaceFile() {
    const file = replaceFileInput.files[0];
    const documentId = pendingReplaceId;
    pendingReplaceId = null;

    if (!file || !documentId) return;
    if (!isPdf(file)) {
        setStatus('Only PDF files are supported.', 'error');
        replaceFileInput.value = '';
        return;
    }

    const adminKey = await requestAdminKey();
    if (!adminKey) {
        replaceFileInput.value = '';
        return;
    }

    documentActionPending = true;
    updateControls();
    setStatus('Uploading replacement...');

    try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(
            `${baseURL}/documents/${encodeURIComponent(documentId)}/replace`,
            {
                method: 'POST',
                headers: { 'X-Admin-Key': adminKey },
                body: formData,
            }
        );
        const data = await readJson(response);
        if (!response.ok) {
            throw new Error(errorMessage(data, 'Document replacement failed'));
        }

        setStatus('Replacement accepted; processing has started.');
        const completed = await waitForJob(data.id, (job) => {
            setStatus(`Replacing document: ${job.phase.replaceAll('_', ' ')} (${job.progress}%)`);
        });
        setStatus(
            completed.result?.status || 'Document replaced successfully.',
            'success'
        );
        await refreshDocuments();
    } catch (error) {
        setStatus(error.message, 'error');
    } finally {
        documentActionPending = false;
        replaceFileInput.value = '';
        updateControls();
    }
}

documentsList.addEventListener('change', (event) => {
    const checkbox = event.target.closest('input[type="checkbox"][data-document-id]');
    if (!checkbox) return;
    if (checkbox.checked) {
        selectedDocumentIds.add(checkbox.dataset.documentId);
    } else {
        selectedDocumentIds.delete(checkbox.dataset.documentId);
    }
    saveFilters();
    updateFilterSummary();
    updateControls();
});

documentsList.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    const documentEntry = documents.find(
        (item) => item.document_id === button.dataset.documentId
    );
    if (!documentEntry) return;
    if (button.dataset.action === 'remove') {
        void removeDocument(documentEntry);
    } else if (button.dataset.action === 'replace') {
        startReplace(documentEntry);
    }
});

uploadFileButton.addEventListener('click', async () => {
    const file = fileInput.files[0];

    if (!file) {
        setStatus('Select a PDF file before uploading.', 'error');
        return;
    }

    if (!isPdf(file)) {
        setStatus('Only PDF files are supported.', 'error');
        return;
    }

    setLoading(uploadLoading, true);
    setStatus('Uploading document...');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${baseURL}/upload-file`, {
            method: 'POST',
            body: formData,
        });
        const data = await readJson(response);

        if (!response.ok) {
            throw new Error(errorMessage(data, 'Upload failed'));
        }

        fileInput.value = '';
        setStatus('Upload accepted; processing has started.');
        const completed = await waitForJob(data.id, (job) => {
            setStatus(`Processing document: ${job.phase.replaceAll('_', ' ')} (${job.progress}%)`);
        });
        setStatus(completed.result?.status || 'Document processed successfully.', 'success');
        await refreshDocuments();
    } catch (error) {
        setStatus(error.message, 'error');
        console.error('Upload failed:', error);
    } finally {
        setLoading(uploadLoading, false);
    }
});

askButton.addEventListener('click', queryRequest);

questionInput.addEventListener('input', updateControls);
questionInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !askButton.disabled) {
        event.preventDefault();
        void queryRequest();
    }
});

clearConversationButton.addEventListener('click', () => {
    if (isStreaming || documentActionPending) return;
    if (!window.confirm('Clear this conversation? Documents will not be removed.')) {
        return;
    }
    exchanges = [];
    sessionId = createSessionId();
    saveConversation();
    renderConversation();
    setStatus('Conversation cleared. Documents were not changed.', 'success');
});

replaceFileInput.addEventListener('change', () => {
    void handleReplaceFile();
});

restartButton.addEventListener('click', async () => {
    let adminKey = null;

    try {
        adminKey = await requestAdminKey();
        if (!adminKey) {
            return;
        }
        setLoading(restartLoading, true);
        restartButton.disabled = true;

        const response = await fetch(`${baseURL}/database`, {
            method: 'DELETE',
            headers: { 'X-Admin-Key': adminKey },
        });
        const data = await readJson(response);

        if (!response.ok) {
            throw new Error(errorMessage(data, 'Clear database failed'));
        }

        fileInput.value = '';
        questionInput.value = '';
        setStatus(data.status || 'Database cleared successfully.', 'success');
        await refreshDocuments();
    } catch (error) {
        setStatus(error.message, 'error');
        console.error('Clear database failed:', error);
        updateControls();
    } finally {
        adminKey = null;
        adminKeyInput.value = '';
        setLoading(restartLoading, false);
    }
});

window.addEventListener('beforeunload', () => {
    if (activeController) {
        activeController.abort();
    }
});

document.addEventListener('DOMContentLoaded', async () => {
    loadConversation();
    loadFilters();
    renderConversation();
    await refreshDocuments();
});

updateControls();
