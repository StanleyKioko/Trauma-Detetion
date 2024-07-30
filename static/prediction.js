document.addEventListener('DOMContentLoaded', function () {
    const predictionForm = document.getElementById('predictionForm');

    if (predictionForm) {
        predictionForm.addEventListener('submit', function (event) {
            event.preventDefault();

            const formData = new FormData(predictionForm);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = value;
            });

            fetch('/submit_prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
            .then(response => response.json())
            .then(result => {
                if (result.status === 'success') {
                    alert('Prediction submitted successfully! Prediction ID: ' + result.prediction_id);
                } else {
                    alert('Error submitting prediction: ' + result.message);
                }
            })
            .catch(error => console.error('Error:', error));
        });
    }
});
