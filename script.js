document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('reviewForm');

    if (form) {
        form.addEventListener('submit', function(event) {
            event.preventDefault();

            const reviewText = document.querySelector('textarea').value;
            if (reviewText.trim() === '') {
                alert('Please enter a review.');
                return;
            }

            const loading = document.getElementById('loading');
            loading.style.display = 'block';

            fetch('/analyze', {
                method: 'POST',
                body: new FormData(form)
            })
            .then(response => response.json()) // Expect JSON response for errors
            .then(data => {
                loading.style.display = 'none';

                if (data.error) {
                    alert(data.error); // Display error message
                } else {
                    // Store result in local storage
                    localStorage.setItem('reviewResult', JSON.stringify(data.result));
                    // Redirect to results page
                    window.location.href = '/result'; 
                }
            })
            .catch(error => {
                console.error("Fetch Error:", error);
                loading.style.display = 'none';
                alert("An error occurred. Check the console.");
            });
        });

        // ... (autosize textarea code - if you have it, paste it here) ...

    } else {
        console.error("Form element NOT found!");
    }
});