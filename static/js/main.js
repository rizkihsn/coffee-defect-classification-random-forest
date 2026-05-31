// =============================================
// CoffeeDefect AI - Main JavaScript
// =============================================

// Toast notification system
function showToast(message, type = 'info') {
    const icons = {
        success: 'bi-check-circle-fill',
        error: 'bi-exclamation-circle-fill',
        warning: 'bi-exclamation-triangle-fill',
        info: 'bi-info-circle-fill'
    };
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast-item ${type}`;
    toast.innerHTML = `<i class="bi ${icons[type] || icons.info}"></i><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'toastIn 0.3s ease reverse forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// Navbar scroll effect
window.addEventListener('scroll', () => {
    const nav = document.getElementById('mainNav');
    if (nav) nav.classList.toggle('scrolled', window.scrollY > 50);
});

// Model status check
async function checkModelStatus() {
    const dot = document.getElementById('modelStatus');
    const text = document.getElementById('modelStatusText');
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.model_loaded) {
            dot.className = 'status-dot online';
            text.textContent = 'Model Ready';
        } else {
            dot.className = 'status-dot offline';
            text.textContent = 'No Model';
        }
    } catch {
        dot.className = 'status-dot offline';
        text.textContent = 'Offline';
    }
}

// Scroll animation observer
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    document.querySelectorAll('.animate-in').forEach(el => observer.observe(el));
}

// Counter animation with prefix & suffix support
function animateCounter(el, target, suffix = '', prefix = '') {
    const duration = 1500;
    const start = 0;
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (target - start) * eased);
        el.textContent = prefix + current + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// Stats Counter Observer
function initCounterAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseInt(el.getAttribute('data-target'), 10) || 0;
                const suffix = el.getAttribute('data-suffix') || '';
                const prefix = el.getAttribute('data-prefix') || '';
                animateCounter(el, target, suffix, prefix);
                observer.unobserve(el); // Animate once
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.stat-number-anim').forEach(el => observer.observe(el));
}

// AOS (Animate On Scroll)
function initAOS() {
    if (typeof AOS === 'undefined') return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    AOS.init({
        duration: 650,
        easing: 'ease-out-cubic',
        once: true,
        offset: 48,
        delay: 0,
        anchorPlacement: 'top-bottom',
    });
}

function refreshAOS() {
    if (typeof AOS !== 'undefined' && typeof AOS.refresh === 'function') {
        AOS.refresh();
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
    initAOS();
    initScrollAnimations();
    initCounterAnimations();
});
