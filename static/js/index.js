'use strict';

const scrollButton = document.getElementById('scroll-to-top');
function updateScrollButton() {
  scrollButton?.classList.toggle('visible', window.scrollY > 300);
}
scrollButton?.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
});
window.addEventListener('scroll', updateScrollButton, { passive: true });
updateScrollButton();

const copyButton = document.getElementById('copy-bibtex');
const copyStatus = document.getElementById('copy-status');
copyButton?.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex-code');
  try {
    if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
    await navigator.clipboard.writeText(citation.textContent);
    copyButton.classList.add('copied');
    copyButton.querySelector('.copy-text').textContent = 'Copied!';
    copyStatus.textContent = 'BibTeX copied to clipboard.';
    setTimeout(() => {
      copyButton.classList.remove('copied');
      copyButton.querySelector('.copy-text').textContent = 'Copy BibTeX';
    }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Press ⌘C or Ctrl+C to copy, or use Download BibTeX.';
  }
});


// Keep every table available without JavaScript; enhance to accessible task tabs.
const resultTabs = document.querySelector('.result-tabs');
if (resultTabs) {
  const tabs = [...resultTabs.querySelectorAll('button')];
  const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));
  function selectResultTab(index, focus = false) {
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
      panels[i].hidden = i !== index;
    });
    if (focus) tabs[index].focus();
  }
  resultTabs.setAttribute('role', 'tablist');
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].tabIndex = 0;
    tab.addEventListener('click', () => selectResultTab(index));
    tab.addEventListener('keydown', event => {
      const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length
        : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
      if (next !== null) {
        event.preventDefault();
        selectResultTab(next, true);
      }
    });
    const control = panels[index].querySelector('.setting-control');
    const select = control.querySelector('select');
    const settings = [...panels[index].querySelectorAll('.result-setting')];
    function selectSetting() {
      settings.forEach(setting => { setting.hidden = setting.id !== select.value; });
    }
    select.addEventListener('change', selectSetting);
    selectSetting();
    control.hidden = false;
  });
  selectResultTab(0);
  resultTabs.hidden = false;
}


// Keep version links pointed at the section being read, including after scrolling.
const versionLinks = [...document.querySelectorAll('[data-version-page]')];
if (versionLinks.length) {
  const readingBar = document.querySelector('.reading-bar');
  const readingSections = [...document.querySelectorAll('main > section[id]')];
  let versionFramePending = false;
  function updateVersionLinks() {
    versionFramePending = false;
    const readingLine = (readingBar?.getBoundingClientRect().height || 0) + 120;
    let sectionId = 'main-content';
    for (const section of readingSections) {
      if (section.getBoundingClientRect().top <= readingLine) sectionId = section.dataset.versionSection || section.id;
      else break;
    }
    versionLinks.forEach(link => {
      link.setAttribute('href', link.dataset.versionPage + '#' + sectionId);
    });
  }
  function scheduleVersionUpdate() {
    if (!versionFramePending) {
      versionFramePending = true;
      requestAnimationFrame(updateVersionLinks);
    }
  }
  // Recheck layout at activation, including after images or equations resize.
  versionLinks.forEach(link => link.addEventListener('click', updateVersionLinks));
  const sectionResizeObserver = new ResizeObserver(scheduleVersionUpdate);
  readingSections.forEach(section => sectionResizeObserver.observe(section));
  window.addEventListener('scroll', scheduleVersionUpdate, { passive: true });
  window.addEventListener('resize', scheduleVersionUpdate);
  window.addEventListener('load', scheduleVersionUpdate);
  window.addEventListener('hashchange', scheduleVersionUpdate);
  updateVersionLinks();
}


// Reference backlinks can point into a closed explanation panel.
function revealCitationTarget() {
  let id;
  try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
  if (!id.startsWith('cite-')) return;
  const target = document.getElementById(id);
  if (!target) return;
  // Table citations may be inside a task tab or setting that is not visible.
  const panel = target.closest('.task-panel');
  if (panel) {
    document.getElementById(panel.getAttribute('aria-labelledby'))?.click();
    const setting = target.closest('.result-setting');
    const selector = panel.querySelector('.setting-control select');
    if (setting && selector) {
      selector.value = setting.id;
      selector.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }
  let parent = target.parentElement;
  while (parent) {
    if (parent.tagName === 'DETAILS') parent.open = true;
    parent = parent.parentElement;
  }
  requestAnimationFrame(() => {
    target.focus({ preventScroll: true });
    target.scrollIntoView({ block: 'start' });
  });
}
window.addEventListener('hashchange', revealCitationTarget);
window.addEventListener('load', revealCitationTarget);


// Track the last heading above the reading line, including long open details.
const pageNav = document.querySelector('.page-nav');
if (pageNav) {
  const toggle = pageNav.querySelector('.page-nav-toggle');
  const currentLabel = pageNav.querySelector('.page-nav-current');
  const entries = [...pageNav.querySelectorAll('a[href^="#"]')].map(link => ({
    link, target: document.getElementById(link.hash.slice(1))
  })).filter(entry => entry.target);
  const desktopNav = window.matchMedia('(min-width: 1200px)');
  let framePending = false;
  let activeLink;
  function closePageMenu() {
    pageNav.classList.remove('is-open');
    toggle.setAttribute('aria-expanded', 'false');
  }
  function updatePageNav() {
    framePending = false;
    const readingLine = desktopNav.matches ? 120 : toggle.getBoundingClientRect().height + 100;
    let current = entries[0];
    for (const entry of entries) {
      if (entry.target.getBoundingClientRect().top <= readingLine) current = entry;
      else break;
    }
    if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) current = entries.at(-1);
    if (!current || current.link === activeLink) return;
    activeLink = current.link;
    entries.forEach(({ link }) => {
      link.removeAttribute('aria-current');
      link.classList.remove('is-parent');
    });
    activeLink.setAttribute('aria-current', 'location');
    currentLabel.textContent = activeLink.textContent;
    const parentItem = activeLink.parentElement.parentElement.closest('li');
    parentItem?.querySelector(':scope > a')?.classList.add('is-parent');
  }
  function schedulePageNav() {
    if (framePending) return;
    framePending = true;
    requestAnimationFrame(updatePageNav);
  }
  pageNav.classList.add('nav-ready');
  toggle.hidden = false;
  toggle.addEventListener('click', () => {
    const open = pageNav.classList.toggle('is-open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  pageNav.addEventListener('keydown', event => {
    if (event.key === 'Escape' && pageNav.classList.contains('is-open')) {
      closePageMenu();
      toggle.focus();
    }
  });
  entries.forEach(({ link }) => link.addEventListener('click', () => {
    if (!desktopNav.matches) closePageMenu();
    schedulePageNav();
  }));
  desktopNav.addEventListener('change', () => { closePageMenu(); schedulePageNav(); });
  window.addEventListener('scroll', schedulePageNav, { passive: true });
  window.addEventListener('resize', schedulePageNav);
  window.addEventListener('hashchange', schedulePageNav);
  window.addEventListener('load', schedulePageNav);
  new ResizeObserver(schedulePageNav).observe(document.getElementById('main-content'));
  updatePageNav();
}

// Switch only the abstract; lay is the initial version.
document.querySelectorAll('[data-summary]').forEach(button => {
  button.addEventListener('click', () => {
    const version = button.dataset.summary;
    document.querySelectorAll('[data-summary]').forEach(option => {
      option.setAttribute('aria-pressed', String(option === button));
    });
    document.getElementById('abstract-technical').hidden = version !== 'technical';
    document.getElementById('abstract-lay').hidden = version !== 'lay';
  });
});
