// BOTGROUND — Global JavaScript

// ─── TOAST NOTIFICATION ───
function showToast(message, type = "success") {
    const toast = document.getElementById("toast");
    if (!toast) return;

    toast.textContent = message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove("show");
    }, 3000);
}

// ─── ACTIVE NAV LINK ───
document.addEventListener("DOMContentLoaded", () => {
    const path = window.location.pathname;
    const links = document.querySelectorAll(".nav-link");

    links.forEach(link => {
        if (link.getAttribute("href") === path) {
            link.classList.add("active");
        } else {
            link.classList.remove("active");
        }
    });
});

// ─── DATE RANGE DEFAULT (Overview and Analytics only, not the Conversations filters) ───
document.addEventListener("DOMContentLoaded", () => {
    const today = new Date().toISOString().split("T")[0];
    const sevenAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
        .toISOString().split("T")[0];

    document.querySelectorAll("#date-form input[name='date_to']").forEach(el => {
        if (!el.value) el.value = today;
    });

    document.querySelectorAll("#date-form input[name='date_from']").forEach(el => {
        if (!el.value) el.value = sevenAgo;
    });
});

// ─── CONFIRM DIALOG ───
function confirmAction(message, callback) {
    if (confirm(message)) callback();
}

// ─── FORMAT NUMBER ───
function formatNumber(num) {
    if (num >= 1000) return (num / 1000).toFixed(1) + "k";
    return num.toString();
}

// ─── KEYBOARD SHORTCUTS ───
document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
        const btnSingle = document.getElementById("btn-single");
        const btnConv = document.getElementById("btn-conv");
        if (btnSingle && !btnSingle.disabled) btnSingle.click();
        if (btnConv && !btnConv.disabled) btnConv.click();
    }
});
// ─── ESCAPE HTML (user text must never be injected as markup) ───
function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
