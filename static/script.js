document.addEventListener('DOMContentLoaded', function () {
    const weatherCard = document.getElementById('weather-card');
    const newsCard = document.getElementById('news-card');
    const quoteCard = document.getElementById('quote-card');
    const commandForm = document.getElementById('command-form');
    const voiceButton = document.getElementById("voice-button");
    const textInput = document.getElementById("text-input");
    const sendButton = document.getElementById("send-button");
    const responseDisplay = document.getElementById("response-display");
    const listeningIndicator = document.getElementById("listening-indicator");
    const clearHistoryButton = document.getElementById("clear-history");
    const themeSwitcher = document.getElementById("theme-selector");
    const themeStyle = document.getElementById("theme-style");
    const volumeControl = document.getElementById("volume-control");
    const reminderModal = document.getElementById('reminder-modal');
    const closeButton = document.querySelector('.close-button');
    const saveReminderButton = document.getElementById('save-reminder');
    const reminderMessageInput = document.getElementById('reminder-message');
    const reminderTimeInput = document.getElementById('reminder-time');
    const setTimerCard = document.getElementById('set-timer-card');
    const calendarCard = document.getElementById('calendar-card');
    const dateDisplay = document.getElementById('date-display');

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition;

    const defaultLanguage = 'en-US';
    let currentLanguage = defaultLanguage;

    // Initialize Speech Recognition
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.lang = currentLanguage;
        recognition.continuous = false;
        recognition.interimResults = false;

        recognition.onstart = function () {
            setListeningIndicator('Listening...', true);
        };

        recognition.onresult = function (event) {
            const command = event.results[0][0].transcript.trim();
            console.log('Recognized Command:', command);
            textInput.value = command; // Set recognized command in input
            sendCommand(command);
            setListeningIndicator('', false);
        };

        recognition.onerror = function (event) {
            console.error('Speech recognition error:', event.error);
            setListeningIndicator(`Error: ${event.error}. Please try again.`, false);
        };

        recognition.onspeechend = function () {
            recognition.stop();
            setListeningIndicator('', false);
        };
    } else {
        console.error('Speech recognition not supported in this browser.');
        setListeningIndicator('Speech recognition not supported in this browser.', false);
    }

    // Helper Functions
    function setListeningIndicator(message, isVisible) {
        if (listeningIndicator) {
            listeningIndicator.style.display = isVisible ? 'block' : 'none';
            listeningIndicator.innerText = message;
        }
    }

    function fetchData(command, loadingMessage) {
        updateResponseDisplay(loadingMessage);
        setListeningIndicator('Loading...', true);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
            handleError(new Error('Request timed out. Please try again.'));
        }, 10000);

        fetch('/voice-command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: command }),
            signal: controller.signal
        })
            .then(handleResponse)
            .catch(handleError)
            .finally(() => {
                clearTimeout(timeoutId);
                setListeningIndicator('', false);
            });
    }

    function handleResponse(response) {
        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.status}`);
        }
        return response.json().then(data => {
            console.log('Fetch Data Response:', data);
            const responseText = data.response || 'No response received.';
            updateResponseDisplay(responseText);
            speakResponse(responseText);
            saveHistory(responseText);
        });
    }

    function handleError(error) {
        console.error('Error fetching data:', error);
        const errorMessage = `An error occurred: ${error.message}`;
        updateResponseDisplay(errorMessage);
        speakResponse(errorMessage);
    }

    function updateResponseDisplay(message) {
        if (responseDisplay) {
            responseDisplay.innerHTML += `<p>${message}</p>`;
            responseDisplay.scrollTop = responseDisplay.scrollHeight;
        }
    }

    function sendCommand(command) {
        console.log('Sending Command:', command);
        fetchData(command, 'Processing command...');
        textInput.value = ""; // Clear input field
    }

    function speakResponse(response) {
        const synth = window.speechSynthesis;
        const volume = volumeControl ? volumeControl.value / 100 : 1;

        if (synth.speaking) {
            synth.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(response);
        utterance.volume = volume; 
        utterance.onend = function () {
            console.log('Speech has finished.');
        };

        synth.speak(utterance);
    }

    function greetUser() {
        const greetingMessage = "Hello! Welcome to your Panda Virtual Assistant. How can I help you today?";
        updateResponseDisplay(greetingMessage);
        speakResponse(greetingMessage);
    }

    function startListening() {
        if (recognition) {
            recognition.lang = currentLanguage;
            recognition.start();
        }
    }

    function saveHistory(message) {
        const chatHistory = JSON.parse(localStorage.getItem('chatHistory')) || [];
        chatHistory.push(message);
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    }

    function loadHistory() {
        const chatHistory = JSON.parse(localStorage.getItem('chatHistory')) || [];
        chatHistory.forEach(msg => updateResponseDisplay(msg));
    }

    function clearHistory() {
        localStorage.removeItem('chatHistory');
        responseDisplay.innerHTML = "";
    }

    // Reminder Modal Functions
    function showReminderModal() {
        reminderModal.style.display = 'flex';
        reminderMessageInput.value = ""; // Clear previous input
        reminderTimeInput.value = ""; // Clear previous input
    }

    function saveReminder() {
        const message = reminderMessageInput.value;
        const time = reminderTimeInput.value;

        if (message && time) {
            const reminders = JSON.parse(localStorage.getItem('reminders')) || [];
            const reminder = { message: message, time: time };
            reminders.push(reminder);
            localStorage.setItem('reminders', JSON.stringify(reminders));

            alert(`Reminder set for "${message}" at ${new Date(time).toLocaleString()}`);
            reminderModal.style.display = 'none';
        } else {
            alert('Please fill in both fields.');
        }
    }

    function showDate() {
        const today = new Date();
        const year = today.getFullYear();
        const month = String(today.getMonth() + 1).padStart(2, '0');  // Months are zero-indexed
        const day = String(today.getDate()).padStart(2, '0');
        
        const formattedDate = `${year}-${month}-${day}`;
        updateResponseDisplay(`Today's date is: ${formattedDate}`);
        speakResponse(`Today's date is: ${formattedDate}`);
    }

    function setTimer() {
        const timerInput = prompt("Enter time in seconds for the timer:");
        const seconds = parseInt(timerInput, 10);
        if (isNaN(seconds) || seconds <= 0) {
            alert("Please enter a valid number of seconds.");
            return;
        }

        updateResponseDisplay(`Timer set for ${seconds} seconds.`);
        speakResponse(`Timer set for ${seconds} seconds.`);
        setTimeout(() => {
            alert("Time's up!");
            updateResponseDisplay("Timer completed.");
            speakResponse("Time's up!");
        }, seconds * 1000);
    }

    function checkCalendarEvents() {
        updateResponseDisplay("Fetching calendar events...");
        speakResponse("Fetching calendar events...");
        setTimeout(() => {
            const events = ["Meeting at 2 PM", "Dinner with family at 6 PM"];
            if (events.length > 0) {
                const eventsMessage = "Your upcoming events are: " + events.join(', ');
                updateResponseDisplay(eventsMessage);
                speakResponse(eventsMessage);
            } else {
                updateResponseDisplay("You have no upcoming events today.");
                speakResponse("You have no upcoming events today.");
            }
        }, 1000);
    }

    // Event Listeners for Cards
    if (weatherCard) {
        weatherCard.addEventListener('click', function () {
            fetchData('weather info', 'Fetching weather info...');
        });
    }

    if (newsCard) {
        newsCard.addEventListener('click', function () {
            fetchData('news updates', 'Fetching news updates...');
        });
    }

    if (quoteCard) {
        quoteCard.addEventListener('click', function () {
            fetchData('quote of the day', 'Fetching quote of the day...');
        });
    }

    // Command Form Submission
    if (commandForm) {
        commandForm.onsubmit = function (event) {
            event.preventDefault();
            const command = textInput.value.trim();
            if (command) {
                sendCommand(command);
            }
        };
    }

    // Voice Command
    if (voiceButton) {
        voiceButton.addEventListener("click", startListening);
    }

    // Send Button
    if (sendButton) {
        sendButton.addEventListener("click", function (event) {
            event.preventDefault();
            const command = textInput.value.trim();
            if (command) {
                sendCommand(command);
            }
        });
    }

    // Clear History Button
    if (clearHistoryButton) {
        clearHistoryButton.addEventListener("click", clearHistory);
    }

    // Theme Switcher
themeSwitcher.addEventListener('change', function () {
    const selectedTheme = themeSwitcher.value;
    
    switch (selectedTheme) {
        case 'styles.css':
            themeStyle.href = 'styles.css'; // Default theme
            break;
        case 'css/theme1.css':
            themeStyle.href = 'static/css/theme1.css'; // Cool Blue theme
            break;
        case 'css/theme2.css':
            themeStyle.href = 'static/css/theme2.css'; // Warm Orange theme
            break;
        case 'css/theme3.css':
            themeStyle.href = 'static/css/theme3.css'; // Vibrant Purple theme
            break;
    }
    
    // Force the browser to reapply the default theme by adding/removing it
    if (selectedTheme === '/static/styles.css') {
        themeStyle.href = 'static/styles.css'; // Reapply default theme
    }
});

    loadHistory();
    greetUser();

    // Volume Control
    if (volumeControl) {
        volumeControl.addEventListener('input', function () {
            const volume = volumeControl.value;
            console.log(`Volume set to: ${volume}`);
            speakResponse(`Volume set to ${volume}`);
        });
    }

    // Reminder Modal
    document.getElementById('set-reminder')?.addEventListener('click', showReminderModal);
    if (closeButton) {
        closeButton.addEventListener('click', function () {
            reminderModal.style.display = 'none';
        });
    }

    if (saveReminderButton) {
        saveReminderButton.addEventListener('click', saveReminder);
    }

    // Set Timer Event
    if (setTimerCard) {
        setTimerCard.addEventListener('click', setTimer);
    }

    // Check Calendar Events Event
    if (calendarCard) {
        calendarCard.addEventListener('click', checkCalendarEvents);
    }

    // Open Browser
    document.getElementById('open-browser')?.addEventListener('click', function () {
        window.open('https://www.google.com', '_blank');
    });

    // Check Weather
    document.getElementById('check-weather')?.addEventListener('click', function () {
        alert('Weather is Sunny.');
    });

    // Play Music
    document.getElementById('play-music')?.addEventListener('click', function () {
        window.open('https://youtu.be/6d5SS0gS5bU?si=S-IAHC8K5Rw060nT', '_blank');
    });

    // Check Time
    document.getElementById('check-time')?.addEventListener('click', function () {
        const currentTime = new Date().toLocaleTimeString();
        alert(`Current time is: ${currentTime}`);
    });

    // Tell Joke
    document.getElementById('tell-joke')?.addEventListener('click', function () {
        const joke = "Why don't scientists trust atoms? Because they make up everything!";
        updateResponseDisplay(joke);
        speakResponse(joke);
    });

    // Show Today's Date
    document.getElementById('show-date')?.addEventListener('click', showDate);
});
 