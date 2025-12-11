// Sets up custom dropdown behavior for all select wrappers
document.addEventListener("DOMContentLoaded", () => {
    const selectWrappers = document.querySelectorAll('.select-wrapper');

    selectWrappers.forEach(wrapper => {
        const selectBox = wrapper.querySelector('.custom-select');
        if (!selectBox) return;

        const selectedText = selectBox.querySelector('.selected');
        const options = selectBox.querySelector('.options');
        const optionList = selectBox.querySelectorAll('.option');

        if (!optionList.length) return;

        const defaultOption = optionList[0];
        selectedText.textContent = defaultOption.textContent;
        defaultOption.classList.add('selected');

        // Toggle options display on select box click
        selectBox.addEventListener('click', () => {
            options.style.display = options.style.display === 'block' ? 'none' : 'block';
            selectBox.classList.toggle('open');
        });

        // Update selected option and hide options on option click
        optionList.forEach(option => {
            option.addEventListener('click', () => {
                selectedText.textContent = option.textContent;
                optionList.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');

                // Show/hide custom prompt textarea
                if (selectBox.id === 'style') {
                    const customWrapper = document.getElementById('custom-prompt-wrapper');
                    if (option.textContent.includes('Custom')) {
                        customWrapper.style.display = 'block';
                    } else {
                        customWrapper.style.display = 'none';
                    }
                }

                // Show/hide translator-specific settings
                if (selectBox.id === 'translator') {
                    const copilotSettings = document.getElementById('copilot-settings');
                    const geminiSettings = document.getElementById('gemini-settings');

                    if (option.textContent === 'Local LLM') {
                        copilotSettings.style.display = 'block';
                        geminiSettings.style.display = 'none';
                    } else if (option.textContent === 'Gemini') {
                        copilotSettings.style.display = 'none';
                        geminiSettings.style.display = 'block';
                    } else {
                        copilotSettings.style.display = 'none';
                        geminiSettings.style.display = 'none';
                    }
                }
            });
        });

        // Hide options when clicking outside the select box
        window.addEventListener('click', e => {
            if (!wrapper.contains(e.target)) {
                options.style.display = 'none';
                selectBox.classList.remove('open');
            }
        });

        // Save dropdown selection to localStorage
        optionList.forEach(option => {
            option.addEventListener('click', () => {
                if (selectBox.id) {
                    localStorage.setItem('select_' + selectBox.id, option.textContent);
                }
            });
        });

        // Restore saved selection on load
        if (selectBox.id) {
            const savedValue = localStorage.getItem('select_' + selectBox.id);
            if (savedValue) {
                optionList.forEach(opt => {
                    if (opt.textContent === savedValue) {
                        selectedText.textContent = savedValue;
                        optionList.forEach(o => o.classList.remove('selected'));
                        opt.classList.add('selected');

                        // Trigger visibility updates for special selects
                        if (selectBox.id === 'style' && savedValue.includes('Custom')) {
                            document.getElementById('custom-prompt-wrapper').style.display = 'block';
                        }
                        if (selectBox.id === 'translator') {
                            const copilotSettings = document.getElementById('copilot-settings');
                            const geminiSettings = document.getElementById('gemini-settings');
                            if (savedValue === 'Local LLM') {
                                copilotSettings.style.display = 'block';
                                geminiSettings.style.display = 'none';
                            } else if (savedValue === 'Gemini') {
                                copilotSettings.style.display = 'none';
                                geminiSettings.style.display = 'block';
                            } else {
                                copilotSettings.style.display = 'none';
                                geminiSettings.style.display = 'none';
                            }
                        }
                    }
                });
            }
        }
    });

    // Load saved Gemini API key from localStorage
    const geminiKeyInput = document.getElementById('gemini_api_key');
    if (geminiKeyInput) {
        const savedKey = localStorage.getItem('gemini_api_key');
        if (savedKey) {
            geminiKeyInput.value = savedKey;
        }
        geminiKeyInput.addEventListener('input', () => {
            localStorage.setItem('gemini_api_key', geminiKeyInput.value);
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

    // Load saved checkbox states from localStorage
    const contextMemoryCheckbox = document.getElementById('context_memory');
    if (contextMemoryCheckbox) {
        const saved = localStorage.getItem('context_memory');
        if (saved !== null) {
            contextMemoryCheckbox.checked = saved === 'true';
        }
        contextMemoryCheckbox.addEventListener('change', () => {
            localStorage.setItem('context_memory', contextMemoryCheckbox.checked);
        });
    }

    const blackBubblesCheckbox = document.getElementById('detect_black_bubbles');
    if (blackBubblesCheckbox) {
        const saved = localStorage.getItem('detect_black_bubbles');
        if (saved !== null) {
            blackBubblesCheckbox.checked = saved === 'true';
        }
        blackBubblesCheckbox.addEventListener('change', () => {
            localStorage.setItem('detect_black_bubbles', blackBubblesCheckbox.checked);
        });
    }
});


// Handles multiple file upload change event
const fileUpload = document.getElementById('file-upload');
if (fileUpload) {
    fileUpload.addEventListener('change', function () {
        const files = this.files;
        const fileList = document.getElementById('file-list');
        const fileText = document.getElementById('file-text');

        if (files.length === 0) {
            fileText.textContent = '📁 Chọn ảnh (có thể chọn nhiều)';
            fileList.innerHTML = '';
            return;
        }

        if (files.length === 1) {
            fileText.textContent = truncateFileName(files[0].name, 25);
            fileList.innerHTML = '';
        } else {
            fileText.textContent = `📁 ${files.length} ảnh đã chọn`;

            // Show file list preview
            fileList.innerHTML = '';
            for (let i = 0; i < Math.min(files.length, 5); i++) {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.textContent = truncateFileName(files[i].name, 30);
                fileList.appendChild(fileItem);
            }

            if (files.length > 5) {
                const moreItem = document.createElement('div');
                moreItem.className = 'file-item more';
                moreItem.textContent = `... và ${files.length - 5} ảnh khác`;
                fileList.appendChild(moreItem);
            }
        }
    });
}

// Truncates file name if it exceeds the maximum length
function truncateFileName(fileName, maxLength) {
    return fileName.length <= maxLength ? fileName : fileName.substr(0, maxLength - 3) + '...';
}

// Updates hidden input fields with selected options
function updateHiddenInputs() {
    const getSelectedText = (id) => {
        const el = document.querySelector(`#${id} .selected`);
        return el ? el.innerText : '';
    };

    document.getElementById("selected_source_lang").value = getSelectedText("source_lang");
    document.getElementById("selected_language").value = getSelectedText("language");
    document.getElementById("selected_translator").value = getSelectedText("translator");
    document.getElementById("selected_style").value = getSelectedText("style");
    document.getElementById("selected_font").value = getSelectedText("font");
    document.getElementById("selected_ocr").value = getSelectedText("ocr");

    // Validate Gemini API key if Gemini is selected
    const translator = getSelectedText("translator");
    if (translator === 'Gemini') {
        const apiKey = document.getElementById('gemini_api_key').value;
        if (!apiKey || apiKey.trim() === '') {
            alert('Vui lòng nhập Gemini API Key!');
            return false;
        }
    }

    // Check if files are selected
    const files = document.getElementById('file-upload').files;
    if (files.length === 0) {
        alert('Vui lòng chọn ít nhất 1 ảnh!');
        return false;
    }

    document.querySelector('form').style.display = 'none';
    document.getElementById('loading-img').style.display = 'block';
    document.getElementById('loading-p').style.display = 'block';

    return true;
}
