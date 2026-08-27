// Sets up custom dropdown behavior for all select wrappers
document.addEventListener("DOMContentLoaded", () => {
    const tt = (key, params) => (window.I18N && I18N.t) ? I18N.t(key, params) : key;

    // Spec 5.6: options carry data-value; display text is translatable and the
    // backend maps the VALUE (additive aliases), never the display text.
    function optionValue(opt) {
        return opt.getAttribute('data-value') || opt.textContent;
    }

    const selectWrappers = document.querySelectorAll('.select-wrapper');

    selectWrappers.forEach(wrapper => {
        const selectBox = wrapper.querySelector('.custom-select');
        if (!selectBox) return;

        const selectedText = selectBox.querySelector('.selected');
        const options = selectBox.querySelector('.options');
        const optionList = selectBox.querySelectorAll('.option');

        if (!optionList.length) return;

        function applyStyleVisibility(val, text) {
            if (selectBox.id !== 'style') return;
            const customWrapper = document.getElementById('custom-prompt-wrapper');
            if (!customWrapper) return;
            customWrapper.style.display = (val === 'custom' || (text || '').indexOf('Custom') >= 0) ? 'block' : 'none';
        }

        function applyTranslatorVisibility(val, text) {
            if (selectBox.id !== 'translator') return;
            const copilotSettings = document.getElementById('copilot-settings');
            const geminiSettings = document.getElementById('gemini-settings');
            const v = String(val || '').toLowerCase();
            if (v === 'copilot' || text === 'Local LLM') {
                copilotSettings.style.display = 'block';
                geminiSettings.style.display = 'none';
            } else if (v === 'gemini' || text === 'Gemini') {
                copilotSettings.style.display = 'none';
                geminiSettings.style.display = 'block';
            } else {
                copilotSettings.style.display = 'none';
                geminiSettings.style.display = 'none';
            }
        }

        const defaultOption = optionList[0];
        selectedText.textContent = defaultOption.textContent;
        defaultOption.classList.add('selected');

        selectBox.addEventListener('click', () => {
            options.style.display = options.style.display === 'block' ? 'none' : 'block';
            selectBox.classList.toggle('open');
        });

        optionList.forEach(option => {
            option.addEventListener('click', () => {
                selectedText.textContent = option.textContent;
                optionList.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');

                applyStyleVisibility(optionValue(option), option.textContent);
                applyTranslatorVisibility(optionValue(option), option.textContent);
            });
        });

        window.addEventListener('click', e => {
            if (!wrapper.contains(e.target)) {
                options.style.display = 'none';
                selectBox.classList.remove('open');
            }
        });

        optionList.forEach(option => {
            option.addEventListener('click', () => {
                if (selectBox.id) {
                    localStorage.setItem('select_' + selectBox.id, optionValue(option));
                }
            });
        });

        if (selectBox.id) {
            const savedValue = localStorage.getItem('select_' + selectBox.id);
            if (savedValue) {
                let matched = null;
                optionList.forEach(opt => {
                    if (optionValue(opt) === savedValue) matched = opt;
                });
                if (!matched) {
                    // A1.4: legacy localStorage stored the display text — migrate
                    // by matching the text, then rewrite the stored value.
                    optionList.forEach(opt => {
                        if (opt.textContent === savedValue) matched = opt;
                    });
                }
                if (matched) {
                    selectedText.textContent = matched.textContent;
                    optionList.forEach(o => o.classList.remove('selected'));
                    matched.classList.add('selected');
                    if (optionValue(matched) !== savedValue) {
                        localStorage.setItem('select_' + selectBox.id, optionValue(matched));
                    }
                    applyStyleVisibility(optionValue(matched), matched.textContent);
                    applyTranslatorVisibility(optionValue(matched), matched.textContent);
                }
            }
        }
    });

    // Load saved Gemini API keys from localStorage
    const geminiKeyInput = document.getElementById('gemini_api_key');
    if (geminiKeyInput) {
        const savedKey = localStorage.getItem('gemini_api_keys') || localStorage.getItem('gemini_api_key');
        if (savedKey) {
            geminiKeyInput.value = savedKey;
        }
        geminiKeyInput.addEventListener('input', () => {
            localStorage.setItem('gemini_api_keys', geminiKeyInput.value);
        });
    }

    // Load saved Gemini model from localStorage
    const geminiModelInput = document.getElementById('gemini_model_input');
    if (geminiModelInput) {
        const savedModel = localStorage.getItem('gemini_model');
        if (savedModel) {
            geminiModelInput.value = savedModel;
        }
        geminiModelInput.addEventListener('input', () => {
            localStorage.setItem('gemini_model', geminiModelInput.value);
        });
    }

    // Load saved Local LLM server URL from localStorage
    const copilotServerInput = document.getElementById('copilot_server');
    if (copilotServerInput) {
        const savedServer = localStorage.getItem('copilot_server');
        if (savedServer) {
            copilotServerInput.value = savedServer;
        }
        copilotServerInput.addEventListener('input', () => {
            localStorage.setItem('copilot_server', copilotServerInput.value);
        });
    }

    // Load saved Local LLM model from localStorage
    const copilotModelInput = document.getElementById('copilot_model_input');
    if (copilotModelInput) {
        const savedModel = localStorage.getItem('copilot_model');
        if (savedModel) {
            copilotModelInput.value = savedModel;
        }
        copilotModelInput.addEventListener('input', () => {
            localStorage.setItem('copilot_model', copilotModelInput.value);
        });
    }

    // Load saved custom prompt from localStorage
    const customPromptInput = document.getElementById('custom_prompt');
    if (customPromptInput) {
        const savedPrompt = localStorage.getItem('custom_prompt');
        if (savedPrompt) {
            customPromptInput.value = savedPrompt;
        }
        customPromptInput.addEventListener('input', () => {
            localStorage.setItem('custom_prompt', customPromptInput.value);
        });
    }
});


// Handles multiple file upload change event
const fileUpload = document.getElementById('file-upload');
if (fileUpload) {
    const tt = (key, params) => (window.I18N && I18N.t) ? I18N.t(key, params) : key;

    function updateFileLabel(files) {
        const fileList = document.getElementById('file-list');
        const fileText = document.getElementById('file-text');
        if (!fileText) return;

        if (!files || files.length === 0) {
            fileText.textContent = tt('index.fileLabel.chooseMany');
            if (fileList) fileList.innerHTML = '';
            return;
        }

        if (files.length === 1) {
            fileText.textContent = truncateFileName(files[0].name, 25);
            if (fileList) fileList.innerHTML = '';
        } else {
            fileText.textContent = tt('index.fileLabel.chosen', { n: files.length });

            if (fileList) fileList.innerHTML = '';
            for (let i = 0; i < Math.min(files.length, 5); i++) {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.textContent = truncateFileName(files[i].name, 30);
                fileList.appendChild(fileItem);
            }

            if (files.length > 5) {
                const moreItem = document.createElement('div');
                moreItem.className = 'file-item more';
                moreItem.textContent = tt('index.fileLabel.more', { n: files.length - 5 });
                fileList.appendChild(moreItem);
            }
        }
    }

    fileUpload.addEventListener('change', function () {
        updateFileLabel(this.files);
    });

    // P1 (A1.3): live-switch hook — re-render the file label on i18n:changed.
    if (window.I18N && I18N.onRefresh) {
        I18N.onRefresh(() => updateFileLabel(fileUpload.files));
    }
}

function truncateFileName(fileName, maxLength) {
    return fileName.length <= maxLength ? fileName : fileName.substr(0, maxLength - 3) + '...';
}

function parseGeminiKeys(rawValue) {
    return (rawValue || '')
        .split(/[\s,;]+/)
        .map(key => key.trim())
        .filter(Boolean);
}

function updateHiddenInputs() {
    const tt = (key, params) => (window.I18N && I18N.t) ? I18N.t(key, params) : key;

    // Spec 5.6: submit the option VALUE (data-value), falling back to the
    // display text for options without data-value (font names).
    const getSelectedValue = (id) => {
        const box = document.querySelector('#' + id);
        if (!box) return '';
        const selected = box.querySelector('.option.selected');
        if (selected) return selected.getAttribute('data-value') || selected.innerText;
        const sel = box.querySelector('.selected');
        return sel ? sel.innerText : '';
    };

    document.getElementById("selected_source_lang").value = getSelectedValue("source_lang");
    document.getElementById("selected_language").value = getSelectedValue("language");
    document.getElementById("selected_translator").value = getSelectedValue("translator");
    document.getElementById("selected_style").value = getSelectedValue("style");
    document.getElementById("selected_font").value = getSelectedValue("font");

    const translator = (getSelectedValue("translator") || '').toLowerCase();
    if (translator === 'gemini') {
        const apiKeys = parseGeminiKeys(document.getElementById('gemini_api_key').value);
        if (apiKeys.length === 0) {
            alert(tt('index.error.noApiKey'));
            return false;
        }

        const modelInput = document.getElementById('gemini_model_input');
        if (!modelInput.value.trim()) {
            alert(tt('index.error.noModel'));
            modelInput.focus();
            return false;
        }
    }

    const files = document.getElementById('file-upload').files;
    if (files.length === 0) {
        alert(tt('index.error.noImages'));
        return false;
    }

    document.querySelector('form').style.display = 'none';
    document.getElementById('loading-img').style.display = 'block';
    document.getElementById('loading-p').style.display = 'block';

    return true;
}
