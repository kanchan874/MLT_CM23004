let chart;

// Simple keyword-based probability model
function analyzeText(text) {

    text = text.toLowerCase();

    let positiveWords = ["love", "amazing", "good", "great", "excellent"];
    let negativeWords = ["hate", "worst", "bad", "terrible"];
    let neutralWords = ["okay", "fine", "average"];

    let posScore = 0;
    let negScore = 0;
    let neuScore = 0;

    positiveWords.forEach(word => {
        if (text.includes(word)) posScore++;
    });

    negativeWords.forEach(word => {
        if (text.includes(word)) negScore++;
    });

    neutralWords.forEach(word => {
        if (text.includes(word)) neuScore++;
    });

    let total = posScore + negScore + neuScore;

    if (total === 0) {
        posScore = negScore = neuScore = 1;
        total = 3;
    }

    return {
        positive: (posScore / total * 100).toFixed(2),
        negative: (negScore / total * 100).toFixed(2),
        neutral: (neuScore / total * 100).toFixed(2)
    };
}

function predict() {

    let text = document.getElementById("inputText").value;

    if (!text.trim()) {
        alert("Please enter a sentence.");
        return;
    }

    let probs = analyzeText(text);

    let maxLabel = Object.keys(probs).reduce((a, b) =>
        probs[a] > probs[b] ? a : b
    );

    document.getElementById("predictionResult").innerText =
        "Prediction: " + maxLabel.toUpperCase();

    updateChart(probs);
}

function updateChart(probs) {

    let ctx = document.getElementById('probChart').getContext('2d');

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                label: 'Prediction Probability (%)',
                data: [
                    probs.positive,
                    probs.negative,
                    probs.neutral
                ]
            }]
        },
        options: {
            animation: {
                duration: 500
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}