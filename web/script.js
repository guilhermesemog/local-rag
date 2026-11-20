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

const baseURL = 'http://127.0.0.1:8000';
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

async function refreshDatabaseState() {
    try {
        const response = await fetch(`${baseURL}/database/`);
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
    setStatus('Uploading and processing document...');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${baseURL}/upload-file/`, {
            method: 'POST',
            body: formData,
        });
        const data = await readJson(response);

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        fileInput.value = '';
        setStatus(data.status || 'File uploaded and processed successfully.', 'success');
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
        const response = await fetch(`${baseURL}/query/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const data = await readJson(response);

        if (!response.ok) {
            throw new Error(data.detail || 'Question failed');
        }

        const title = document.createElement('h3');
        title.textContent = 'Answer:';

        const paragraph = document.createElement('p');
        paragraph.textContent = String(data.results ?? 'No answer was returned.');

        const sourceList = document.createElement('ul');
        for (const source of data.sources ?? []) {
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
        const response = await fetch(`${baseURL}/database/`, {
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
