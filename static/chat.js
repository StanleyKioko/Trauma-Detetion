document.addEventListener('DOMContentLoaded', function () {
    const messageForm = document.getElementById('messageForm');
    const chatContainer = document.getElementById('chat');

    if (messageForm) {
        const socket = io.connect('http://' + document.domain + ':' + location.port);

        messageForm.addEventListener('submit', function (event) {
            event.preventDefault();

            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            const recipient_id = messageForm.getAttribute('data-recipient-id'); // Add recipient ID to the form

            if (message) {
                socket.emit('send_message', {user_id: user_id, message: message, recipient_id: recipient_id});

                messageInput.value = '';
            }
        });

        socket.on('receive_message', function (data) {
            const chatMessage = document.createElement('div');
            chatMessage.textContent = `${data.timestamp} - ${data.sender_id}: ${data.message}`;
            chatContainer.appendChild(chatMessage);
        });

        socket.on('status', function (data) {
            const statusMessage = document.createElement('div');
            statusMessage.textContent = data.msg;
            chatContainer.appendChild(statusMessage);
        });
    }
});
