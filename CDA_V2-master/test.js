document.addEventListener("DOMContentLoaded", function () {
    var next_click = document.querySelectorAll(".next_button");
    var main_form = document.querySelectorAll(".main");
    var step_list = document.querySelectorAll(".progress-bar li");
    var num = document.querySelector(".step-number");
    let formnumber = 0;

    const newProjectInput = document.getElementById('newProjectInput');
    const existingProjectSelect = document.getElementById('existingProjectSelect');
    const fileInput = document.getElementById('file-upload');
    const fileList = document.getElementById('fileList');
    const autoFixButton = document.getElementById("autoFixButton");

    // Disable/Enable inputs based on user selection
    newProjectInput.addEventListener('input', () => {
        if (newProjectInput.value.trim() !== '') {
            existingProjectSelect.disabled = true;
            next_click[formnumber].disabled = false;
        } else {
            existingProjectSelect.disabled = false;
            next_click[formnumber].disabled = existingProjectSelect.value === '';
        }
    });

    existingProjectSelect.addEventListener('change', () => {
        if (existingProjectSelect.value !== '') {
            newProjectInput.disabled = true;
            next_click[formnumber].disabled = false;
        } else {
            newProjectInput.disabled = false;
            next_click[formnumber].disabled = newProjectInput.value.trim() === '';
        }
    });

    // Update file list and enable/disable next button
    fileInput.addEventListener('change', () => {
        fileList.innerHTML = '';
        const files = fileInput.files;

        if (files.length > 0) {
            const fileNames = Array.from(files).map(file => file.name).join(', ');
            fileList.textContent = fileNames;
            next_click[1].disabled = false;
        } else {
            next_click[1].disabled = true;
        }
    });

    next_click.forEach(function (next_click_form) {
        next_click_form.addEventListener('click', function () {
            if (!validateform()) {
                return false;
            }
            formnumber++;
            updateform();
            progress_forward();
            contentchange();

            // Call the appropriate backend function based on the current step
            switch (formnumber) {
                case 1:
                    createDatabase();
                    break;
                case 2:
                    uploadFilesAndGetInfo();
                    break;
                case 3:
                    // File Stats Step: File stats are already updated in uploadFilesAndGetInfo()
                    break;
                case 4:
                    sanitizeData();
                    break;
                case 5:
                    createTablesAndInsertData();
                    break;
                default:
                    break;
            }
        });
    });

    var back_click = document.querySelectorAll(".back_button");
    back_click.forEach(function (back_click_form) {
        back_click_form.addEventListener('click', function () {
            if (formnumber > 0) {
                formnumber--;
                updateform();
                progress_backward();
                contentchange();
            }
        });
    });

    var submit_click = document.querySelectorAll(".submit_button");
    submit_click.forEach(function (submit_click_form) {
        submit_click_form.addEventListener('click', function () {
            if (!validateform()) {
                return false;
            }
            formnumber++;
            updateform();
        });
    });

    function updateform() {
        main_form.forEach(function (mainform_number, index) {
            mainform_number.classList.remove('active');
            if (index === formnumber) {
                mainform_number.classList.add('active');
            }
        });
    }

    function progress_forward() {
        num.innerHTML = formnumber + 1;
        if (formnumber < step_list.length) {
            step_list[formnumber].classList.add('active');
        }
    }

    function progress_backward() {
        var form_num = formnumber + 1;
        if (form_num < step_list.length) {
            step_list[form_num].classList.remove('active');
        }
        num.innerHTML = form_num;
        if (form_num === 5) {
            num.innerHTML = "Summary";
        }
    }

    var step_num_content = document.querySelectorAll(".step-number-content");
    function contentchange() {
        step_num_content.forEach(function (content) {
            content.classList.remove('active');
            content.classList.add('d-none');
        });
        if (formnumber < step_num_content.length) {
            step_num_content[formnumber].classList.add('active');
        }
    }

    function validateform() {
        let validate = true;
        var validate_inputs = document.querySelectorAll(".main.active input[required], .main.active select[required]");
        validate_inputs.forEach(function (validate_input) {
            validate_input.classList.remove('warning');
            if (validate_inputs.length === 2 && validate_input.value.length === 0 && validate_inputs[0].value.length === 0 && validate_inputs[1].value.length === 0) {
                validate = false;
                validate_input.classList.add('warning');
            }
        });
        return validate;
    }

    const inputDiv = document.querySelector('.input-div');
    inputDiv.addEventListener('dragover', (e) => {
        e.preventDefault();
        inputDiv.style.backgroundColor = '#f0f0f0';
    });
    inputDiv.addEventListener('dragleave', () => {
        inputDiv.style.backgroundColor = 'transparent';
    });
    inputDiv.addEventListener('drop', (e) => {
        e.preventDefault();
        inputDiv.style.backgroundColor = 'transparent';
        const files = e.dataTransfer.files;
        fileInput.files = files;
    });

    // Update file statistics in step 3
    fileInput.addEventListener('change', () => {
        const files = fileInput.files;
        const fileNames = Array.from(files).map(file => file.name).join(', ');
        const totalFileSize = Array.from(files).reduce((total, file) => total + file.size, 0);
        const totalRows = 0; // Replace with actual row count
        const totalColumns = 0; // Replace with actual column count
        document.getElementById('file_names').textContent = fileNames;
        document.getElementById('total_file_size').textContent = `${formatFileSize(totalFileSize)}`;
        document.getElementById('total_rows').textContent = totalRows;
        document.getElementById('total_columns').textContent = totalColumns;
    });

    // Helper function to format file size
    function formatFileSize(bytes) {
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let size = bytes;
        let unitIndex = 0;
        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }
        return `${size.toFixed(2)} ${units[unitIndex]}`;
    }

    

    function updateFileStatistics(fileInfo) {
        const fileDetailsContainer = document.getElementById('file_details_container');
        fileDetailsContainer.innerHTML = ''; // Clear existing content

        fileInfo.forEach(info => {
            const fileDetails = document.createElement('div');
            fileDetails.classList.add('file-details');

            fileDetails.innerHTML = `
                <p>File Name: <span>${info.filename}</span></p>
                <p>File Size: <span>${info["file_size(MB)"].toFixed(2)} MB</span></p>
                <p>Total Rows: <span>${info.total_rows}</span></p>
                <p>Total Columns: <span>${info.total_columns}</span></p>
                <br>
            `;

            fileDetailsContainer.appendChild(fileDetails);
        });
    }

    async function createDatabase() {
        const newProjectName = newProjectInput.value.trim();
        const existingProjectName = existingProjectSelect.value;

        let projectName = "";

        if (newProjectName) {
            projectName = newProjectName;
        } else if (existingProjectName) {
            projectName = existingProjectName;
        } else {
            console.error("Project name is required.");
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:8000/create_db', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ db_name: projectName })
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error("Error creating database:", errorData);
                return;
            }

            const data = await response.json();
            console.log(data);
        } catch (error) {
            console.error("Failed to fetch:", error);
        }
    }
    async function uploadFilesAndGetInfo() {
        const formData = new FormData();
        for (const file of fileInput.files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch('http://127.0.0.1:8000/upload_file_info', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error("Failed to upload files and get info.");
            }

            const data = await response.json();
            console.log(data);
            if (data.file_info) {
                updateFileStatistics(data.file_info);
            }
        } catch (error) {
            console.error("Failed to fetch:", error);
        }
    }
    async function sanitizeData() {
        try {
            const response = await fetch('http://127.0.0.1:8000/upload_and_clean', {
                method: 'POST'
            });
    
            if (!response.ok) {
                throw new Error('Failed to sanitize data.');
            }
    
            const data = await response.json();
            console.log(data);
    
            displaySanitizationResults(data.sanitization_infos);
        } catch (error) {
            console.error("Failed to fetch:", error);
        }
    }
    

    document.addEventListener('DOMContentLoaded', () => {
        // Your code here
        async function sanitizeData() {
            const myButton = document.getElementById('myButton');
            if (myButton) {
                myButton.style.display = 'none';
    
                try {
                    const response = await fetch('http://127.0.0.1:8000/upload_and_clean', {
                        method: 'POST'
                    });
    
                    if (!response.ok) {
                        throw new Error('Failed to sanitize data.');
                    }
    
                    const data = await response.json();
                    console.log(data);
    
                    displaySanitizationResults(data);
                } catch (error) {
                    console.error("Failed to fetch:", error);
                }
            } else {
                console.error("Element with id 'myButton' not found.");
            }
        }
    
        sanitizeData(); // Call your function here or wherever it's needed
    });
    
    function displaySanitizationResults(data) {
        const resultContainer = document.getElementById('sanitizationResults');
        resultContainer.innerHTML = ''; // Clear existing content
    
        data.forEach(file => {
            const fileResult = document.createElement('div');
            fileResult.classList.add('file-result');
    
            fileResult.innerHTML = `
                <p><strong>File Name:</strong> ${file.filename}</p>
                <p><strong>Original Shape:</strong> ${file.original_shape.join(', ')}</p>
                <p><strong>Column Names Sanitized:</strong> ${file.column_names_sanitized}</p>
                <p><strong>Special Characters Removed from Column Names:</strong> ${file.special_characters_removed_from_column_names}</p>
                <p><strong>Whitespace Removed:</strong> ${file.whitespace_removed}</p>
                <p><strong>Dates Standardized:</strong> ${file.dates_standardized}</p>
                <p><strong>Missing Values Filled:</strong> ${file.missing_values_filled}</p>
                <p><strong>Duplicates Removed:</strong> ${file.duplicates_removed}</p>
                <hr>
            `;
    
            resultContainer.appendChild(fileResult);
        });
    }
    

    async function createTablesAndInsertData() {
        try {
            const response = await fetch('http://127.0.0.1:8000/create_tables_with_relationships', {
                method: 'POST'
            });
            if (!response.ok) {
                throw new Error('Failed to create tables with relationships');
            }
            const data = await response.json();
            console.log(data);
            // Add any UI update or success message here
            alert("Data has been successfully loaded into the Database!");
        } catch (error) {
            console.error('Error:', error);
            // Handle errors here
            alert("An error occurred while creating tables with relationships.");
        }
    }

    // Event listener for Auto Fix button
    autoFixButton.addEventListener("click", function () {
        sanitizeData();
        autoFixButton.style.display = "none"; // Hide the "Auto Fix" button
        nextStepButton.disabled = false; // Enable the "Next Step" button
    });

});
