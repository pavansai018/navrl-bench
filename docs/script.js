const themeSelect = document.getElementById("themeSelect");
const savedTheme = localStorage.getItem("navrl-theme") || "modern";

document.documentElement.setAttribute("data-theme", savedTheme);
themeSelect.value = savedTheme;

themeSelect.addEventListener("change", function () {
  const selectedTheme = themeSelect.value;
  document.documentElement.setAttribute("data-theme", selectedTheme);
  localStorage.setItem("navrl-theme", selectedTheme);
});
