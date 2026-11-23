const askButton = document.getElementById('askButton');
const restartButton = document.getElementById('restartButton');
const uploadFileButton = document.getElementById('uploadFileButton');

const fileInput = document.getElementById('fileInput');
const questionInput = document.getElementById('questionInput');

const answerContent = document.getElementById('answerContent');

const uploadLoading = document.getElementById('uploadLoading');
const answerLoading = document.getElementById('answerLoading');
const restartLoading = document.getElementById('restartLoading');

const emptyMessage = document.getElementById('emptyMessage');
const appStatus = document.getElementById('appStatus');

const baseURL = window.location.origin;
let databaseIsEmpty = true;

function setStatus(message, type = '') {
    appStatus.textContent = message;
    appStatus.className = `status${type ? ` ${type}` : ''}`;
}

function setLoading(element, isLoading) {
    element.classList.toggle('hidden', !isLoading);
}

function updateControls() {
    restartButton.disabled = databaseIsEmpty;
    askButton.disabled = databaseIsEmpty || questionInput.value.trim() === '';
}

async function readJson(response) {
    try {
        return await response.json();
    } catch {
        return {};
    }
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

    setLoading(answerLoading, true);
    askButton.disabled = true;
    answerContent.replaceChildren();

    try {
        const response = await fetch(`${baseURL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const data = await readJson(response);

        if (!response.ok) {
            throw new Error(data.detail || 'Question failed');
        }

        setStatus('Question accepted; generating an answer...');
        const completed = await waitForJob(data.id, (job) => {
            setStatus(`Answering: ${job.phase.replaceAll('_', ' ')} (${job.progress}%)`);
        });
        const answer = completed.result || {};

        const title = document.createElement('h3');
        title.textContent = 'Answer:';

        const paragraph = document.createElement('p');
        paragraph.textContent = String(answer.results ?? 'No answer was returned.');

        const sourceList = document.createElement('ul');
        for (const source of answer.sources ?? []) {
            const item = document.createElement('li');
            const label = document.createElement('strong');
            const page = source.page ? `, page ${source.page}` : '';
            label.textContent = `${source.document_name || 'Unknown document'}${page}`;
            item.append(label);

            if (source.text) {
                const excerpt = document.createElement('p');
                excerpt.textContent = source.text;
                item.append(excerpt);
            }
            sourceList.append(item);
        }

        const sourcesHeading = document.createElement('h4');
        sourcesHeading.textContent = 'Sources';
        answerContent.replaceChildren(
            title,
            paragraph,
            ...(sourceList.childElementCount ? [sourcesHeading, sourceList] : []),
        );
        questionInput.value = '';
    } catch (error) {
        setStatus(error.message, 'error');
        console.error('Question failed:', error);
    } finally {
        setLoading(answerLoading, false);
        updateControls();
    }
}

askButton.addEventListener('click', queryRequest);

questionInput.addEventListener('input', updateControls);
questionInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !askButton.disabled) {
        event.preventDefault();
        void queryRequest();
    }
});

restartButton.addEventListener('click', async () => {
    setLoading(restartLoading, true);
    restartButton.disabled = true;

    try {
        const response = await fetch(`${baseURL}/database`, {
            method: 'DELETE',
        });
        const data = await readJson(response);

        if (!response.ok) {
            throw new Error(data.detail || 'Clear database failed');
        }

        fileInput.value = '';
        questionInput.value = '';
        answerContent.replaceChildren();
        setStatus(data.status || 'Database cleared successfully.', 'success');
        await refreshDatabaseState();
    } catch (error) {
        setStatus(error.message, 'error');
        console.error('Clear database failed:', error);
        updateControls();
    } finally {
        setLoading(restartLoading, false);
    }
});

document.addEventListener('DOMContentLoaded', refreshDatabaseState);
