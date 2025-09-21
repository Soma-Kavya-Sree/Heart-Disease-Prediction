// static/js/script.js
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("predict-form");
  const predictBtn = document.getElementById("predict-btn");
  const spinner = document.getElementById("spinner");
  const resultArea = document.getElementById("result-area");
  const predictionText = document.getElementById("prediction-text");
  const probText = document.getElementById("prob-text");
  const shapImg = document.getElementById("shap-img");
  const limeLink = document.getElementById("lime-link");
  const resetBtn = document.getElementById("reset-btn");

  function showSpinner(show) {
    if (show) {
      spinner.classList.remove("hidden");
      predictBtn.disabled = true;
    } else {
      spinner.classList.add("hidden");
      predictBtn.disabled = false;
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    showSpinner(true);

    // collect inputs in order
    const inputs = [];
    for (let i = 0; i < FEATURE_COUNT; i++) {
      const el = form.querySelector(`input[name="f${i}"]`);
      if (!el || el.value.trim() === "") {
        alert("Please fill all fields");
        showSpinner(false);
        return;
      }
      inputs.push(Number(el.value));
    }

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs: inputs }),
      });
      const data = await res.json();
      if (data.status !== "ok") {
        alert("Error: " + (data.message || data.error || JSON.stringify(data)));
        showSpinner(false);
        return;
      }

      const result = data.result;
      // show result area
      resultArea.classList.remove("hidden");
      predictionText.textContent = result.prediction === 1 ? "⚠️ Likely HEART DISEASE" : "✅ Likely HEALTHY";
      probText.textContent = `Probabilities — Healthy: ${result.probabilities.healthy.toFixed(4)}, HeartDisease: ${result.probabilities.heart_disease.toFixed(4)}`;
      shapImg.src = result.shap_url + "?v=" + Date.now(); // cache-bust
      limeLink.href = result.lime_url;

    } catch (err) {
      console.error(err);
      alert("Server error. See console.");
    } finally {
      showSpinner(false);
    }
  });

  resetBtn.addEventListener("click", () => {
    // hide result and clear inputs
    resultArea.classList.add("hidden");
    const inputsEls = form.querySelectorAll("input");
    inputsEls.forEach((el) => (el.value = ""));
  });
});
