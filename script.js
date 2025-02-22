document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("queryForm").addEventListener("submit", function (e) {
        e.preventDefault();

        let query = document.getElementById("queryInput").value;

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: "query=" + encodeURIComponent(query)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById("result").innerText = "Error: " + data.error;
            } else {
                let output = `
                    <p><strong>Section:</strong> ${data.data.Section}</p>
                    <p><strong>Offense:</strong> ${data.data.Offense}</p>
                    <p><strong>Punishment:</strong> ${data.data.Punishment}</p>
                    <p><strong>Case Type:</strong> ${data.data["Case Type"]}</p>
                    <p><strong>Procedure:</strong> ${data.data.Procedure}</p>
                `;
                document.getElementById("result").innerHTML = output;

                document.getElementById("downloadBtn").style.display = "block";
                document.getElementById("downloadBtn").onclick = function () {
                    window.location.href = "/download_report";
                };
            }
        });
    });
});
