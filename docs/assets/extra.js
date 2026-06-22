let isHeroInitialized = false;
let scrollListener = null;
let originalColorScheme = null;

function onScroll() {
  const bgLayer = document.querySelector(".parallax-bg");
  const fgLayer = document.querySelector(".parallax-fg");
  const scrollY = window.scrollY;

  // Handle header transparency
  if (scrollY > 50) {
    document.body.classList.remove("is-home-top");
  } else {
    document.body.classList.add("is-home-top");
  }

  // Handle Parallax
  if (bgLayer) {
    bgLayer.style.transform = `translateY(${scrollY * 0.5}px)`;
  }
  if (fgLayer) {
    fgLayer.style.transform = `translateY(${scrollY * 0.15}px)`;
  }
}

function initHero() {
  const heroContainer = document.getElementById("hero-container");

  // Clean up previous state if any
  if (scrollListener) {
    window.removeEventListener("scroll", scrollListener);
    scrollListener = null;
  }
  document.body.classList.remove("is-home-top");
  document.body.classList.remove("is-home-page");

  // Restore the original color scheme if we are navigating away from the home page
  if (originalColorScheme) {
    document.body.setAttribute("data-md-color-scheme", originalColorScheme);
    originalColorScheme = null;
  }

  if (heroContainer) {
    document.body.classList.add("is-home-top");
    document.body.classList.add("is-home-page");

    // Force the home page into dark mode, saving the user's preference to restore later
    const currentScheme = document.body.getAttribute("data-md-color-scheme");
    if (currentScheme !== "slate") {
      originalColorScheme = currentScheme;
      document.body.setAttribute("data-md-color-scheme", "slate");
    }

    scrollListener = onScroll;
    window.addEventListener("scroll", scrollListener, { passive: true });
    
    // Initial call
    onScroll();
  }
}

// MkDocs Material SPA observable fires on load and navigation
if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    initHero();
  });
} else {
  // Fallback for non-SPA
  if (document.readyState === 'loading') {
    document.addEventListener("DOMContentLoaded", initHero);
  } else {
    initHero();
  }
}
