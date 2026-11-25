const askButton = document.getElementById('askButton');
const restartButton = document.getElementById('restartButton');
const uploadFileButton = document.getElementById('uploadFileButton');

const fileInput = document.getElementById('fileInput');
const questionInput = document.getElementById('questionInput');

const answerElement = document.getElementById('answer');
const answerContent = document.getElementById('answerContent');
const queryProgress = document.getElementById('queryProgress');
const queryProgressBar = document.getElementById('queryProgressBar');
const queryPhase = document.getElementById('queryPhase');
const copyRow = document.getElementById('copyRow');
const copyAnswerButton = document.getElementById('copyAnswerButton');
const sourcesSection = document.getElementById('sourcesSection');
const sourcesList = document.getElementById('sourcesList');
const errorMessage = document.getElementById('errorMessage');

const uploadLoading = document.getElementById('uploadLoading');
const answerLoading = document.getElementById('answerLoading');
const restartLoading = document.getElementById('restartLoading');

const emptyMessage = document.getElementById('emptyMessage');
const appStatus = document.getElementById('appStatus');
const adminKeyDialog = document.getElementById('adminKeyDialog');
const adminKeyInput = document.getElementById('adminKeyInput');

const baseURL = window.location.origin;
const PHASE_LABELS = {
    checking_index: 'Checking the document index…',
    searching_documents: 'Retrieving relevant context…',
    generating_answer: 'Generating the answer…',
    formatting_result: 'Finishing the answer…',
    completed: 'Answer ready.',
};

let databaseIsEmpty = true;
let currentAnswer = '';
let activeController = null;
let isStreaming = false;

function setStatus(message, type = '') {
    appStatus.textContent = message;
    appStatus.className = `status${type ? ` ${type}` : ''}`;
}

function setLoading(element, isLoading) {
    element.classList.toggle('hidden', !isLoading);
}

function updateControls() {
    restartButton.disabled = databaseIsEmpty || isStreaming;
    askButton.disabled = databaseIsEmpty || isStreaming || questionInput.value.trim() === '';
}

async function readJson(response) {
    try {
        return await response.json();
    } catch {
        return {};
    }
}

function clearError() {
    errorMessage.textContent = '';
    errorMessage.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
    queryPhase.textContent = 'Answer failed.';
    answerElement.setAttribute('aria-busy', 'false');
    setLoading(answerLoading, false);
}

function resetAnswer() {
    currentAnswer = '';
    answerContent.replaceChildren();
    sourcesList.replaceChildren();
    sourcesSection.classList.add('hidden');
    copyRow.classList.add('hidden');
    copyAnswerButton.textContent = 'Copy answer';
    clearError();
    queryProgress.classList.add('hidden');
    queryProgressBar.value = 0;
    queryPhase.textContent = '';
    answerElement.setAttribute('aria-busy', 'false');
}

function showProgress() {
    queryProgress.classList.remove('hidden');
    answerElement.setAttribute('aria-busy', 'true');
}

function setProgress(phase, progress) {
    const value = Number(progress);
    queryProgress.classList.remove('hidden');
    queryProgressBar.value = Number.isFinite(value)
        ? Math.max(0, Math.min(100, value))
        : 0;
    queryPhase.textContent = PHASE_LABELS[phase] || 'Processing…';
}

function appendToken(text) {
    if (!text) return;
    setLoading(answerLoading, false);
    currentAnswer += text;
    answerContent.append(document.createTextNode(text));
}

function renderSources(sources) {
    sourcesList.replaceChildren();
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

        sourcesList.append(item);
    }

    sourcesSection.classList.toggle('hidden', sourcesList.childElementCount === 0);
}

function finishAnswer(results, sources) {
    const text = String(results ?? '').trim();
    currentAnswer = text;
    answerContent.textContent = text || 'No answer was returned.';
    renderSources(sources || []);
    copyRow.classList.toggle('hidden', !text);
    queryProgress.classList.remove('hidden');
    queryProgressBar.value = 100;
    queryPhase.textContent = 'Answer ready.';
    answerElement.setAttribute('aria-busy', 'false');
    setLoading(answerLoading, false);
    questionInput.value = '';
    updateControls();
}

async function copyAnswer() {
    if (!currentAnswer) return;

    let copied = false;
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(currentAnswer);
            copied = true;
        }
    } catch {
        copied = false;
    }

    if (!copied) {
        const textarea = document.createElement('textarea');
        textarea.value = currentAnswer;
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
        showError('The answer could not be copied.');
        return;
    }

    const previousLabel = copyAnswerButton.textContent;
    copyAnswerButton.textContent = 'Copied!';
    window.setTimeout(() => {
        copyAnswerButton.textContent = previousLabel;
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
            throw new Error(job.detail || 'Failed to check processing status');
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
        if (job.result && Array.isArray(job.result.sources)) {
            renderSources(job.result.sources);
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
    resetAnswer();
    showProgress();
    setProgress('checking_index', 0);
    setLoading(answerLoading, true);

    let reader = null;
    let jobId = null;
    let sawDone = false;

    try {
        const response = await fetch(`${baseURL}/query/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
            signal: controller.signal,
        });

        if (!response.ok) {
            const data = await readJson(response);
            throw new Error(data.detail || 'Question failed');
        }

        if (!response.body) {
            const fallback = await fetch(`${baseURL}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question }),
                signal: controller.signal,
            });
            const data = await readJson(fallback);
            if (!fallback.ok) {
                throw new Error(data.detail || 'Question failed');
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
                    renderSources(data.sources || []);
                } else if (event === 'token') {
                    appendToken(data.text || '');
                } else if (event === 'done') {
                    sawDone = true;
                    finishAnswer(data.results, data.sources);
                } else if (event === 'error') {
                    throw new Error(data.detail || 'The answer could not be generated.');
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
        isStreaming = false;
        activeController = null;
        setLoading(answerLoading, false);
        updateControls();
    }
}

async function queryRequest() {
    const question = questionInput.value.trim();

    if (!question) {
        showError('Enter a question before submitting.');
        return;
    }

    if (databaseIsEmpty) {
        showError('Upload a document before asking a question.');
        return;
    }

    if (isStreaming) return;

    clearError();
    await streamQuery(question);
}

async function refreshDatabaseState() {
    try {
        const response = await fetch(`${baseURL}/database`);
        const data = await readJson(response);

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to check database');
        }

        databaseIsEmpty = Boolean(data.is_empty);
        emptyMessage.textContent = databaseIsEmpty
            ? 'No documents uploaded yet.'
            : 'Documents are ready for questions.';
        updateControls();
    } catch (error) {
        setStatus(error.message, 'error');
        databaseIsEmpty = true;
        updateControls();
    }
}

uploadFileButton.addEventListener('click', async () => {
    const file = fileInput.files[0];

    if (!file) {
        setStatus('Select a PDF file before uploading.', 'error');
        return;
    }

    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
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
            throw new Error(data.detail || 'Upload failed');
        }

        fileInput.value = '';
        setStatus('Upload accepted; processing has started.');
        const completed = await waitForJob(data.id, (job) => {
            setStatus(`Processing document: ${job.phase.replaceAll('_', ' ')} (${job.progress}%)`);
        });
        setStatus(completed.result?.status || 'Document processed successfully.', 'success');
        await refreshDatabaseState();
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

copyAnswerButton.addEventListener('click', copyAnswer);

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
            throw new Error(data.detail || 'Clear database failed');
        }

        resetAnswer();
        fileInput.value = '';
        questionInput.value = '';
        setStatus(data.status || 'Database cleared successfully.', 'success');
        await refreshDatabaseState();
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

document.addEventListener('DOMContentLoaded', refreshDatabaseState);
updateControls();
