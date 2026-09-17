function copyBibTeX() {
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    navigator.clipboard.writeText(document.getElementById('bibtex-code').textContent).then(function() {
        button.classList.add('copied');
        copyText.textContent = 'Copied';
        setTimeout(function() {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    });
}

function scrollToTop() {
    window.scrollTo({top: 0, behavior: 'smooth'});
}

window.addEventListener('scroll', function() {
    document.querySelector('.scroll-to-top').classList.toggle('visible', window.scrollY > 300);
});
