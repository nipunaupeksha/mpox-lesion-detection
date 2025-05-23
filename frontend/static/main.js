/*=============== SHOW MENU ===============*/
const navMenu = document.getElementById("nav-menu"),
  navToggle = document.getElementById("nav-toggle"),
  navClose = document.getElementById("nav-close");

/*==== MENU SHOW =====*/
if (navToggle) {
  navToggle.addEventListener("click", () => {
    navMenu.classList.add("show-menu");
  });
}

/*==== MENU HIDDEN ====*/
if (navClose) {
  navClose.addEventListener("click", () => {
    navMenu.classList.remove("show-menu");
  });
}

/*=============== REMOVE MENU MOBILE ===============*/
const navLink = document.querySelectorAll(".nav__link");

function linkAction() {
  const navMenu = document.getElementById("nav-menu");
  navMenu.classList.remove("show-menu");
}

navLink.forEach((n) => n.addEventListener("click", linkAction));

/*=============== CHANGE BACKGROUND HEADER ===============*/
function scrollHeader() {
  const header = document.getElementById("header");
  if (this.scrollY >= 50) header.classList.add("scroll-header");
  else header.classList.remove("scroll-header");
}
window.addEventListener("scroll", scrollHeader);

/*=============== SCROLL SECTIONS ACTIVE LINK ===============*/
const sections = document.querySelectorAll("section[id], footer[id]");

function scrollActive() {
  const scrollY = window.pageYOffset;
  const windowHeight = window.innerHeight;
  const bodyHeight = document.body.offsetHeight;

  // First, remove all active-link classes
  document
    .querySelectorAll(".nav__menu a")
    .forEach((link) => link.classList.remove("active-link"));

  // Special case: if we're at the bottom of the page, activate footer and return
  if (windowHeight + scrollY >= bodyHeight - 2) {
    const footerLink = document.querySelector(".nav__menu a[href*='footer']");
    if (footerLink) footerLink.classList.add("active-link");
    return;
  }

  // Otherwise, activate section in view
  sections.forEach((current) => {
    const sectionHeight = current.offsetHeight;
    const sectionTop = current.offsetTop - 58;
    const sectionId = current.getAttribute("id");

    if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
      const link = document.querySelector(`.nav__menu a[href*="${sectionId}"]`);
      if (link) link.classList.add("active-link");
    }
  });
}

window.addEventListener("scroll", scrollActive);

/*=============== DARK LIGHT THEME ===============*/
const themeButton = document.getElementById("theme-button");
const darkTheme = "dark-theme";
const iconTheme = "ri-sun-line";

// Previously selected topic (if user selected)
const selectedTheme = localStorage.getItem("selected-theme");
const selectedIcon = localStorage.getItem("selected-icon");

// We obtain the current theme that the interface has by validating the dark-theme class
const getCurrentTheme = () =>
  document.body.classList.contains(darkTheme) ? "dark" : "light";
const getCurrentIcon = () =>
  themeButton.classList.contains(iconTheme) ? "ri-moon-line" : "ri-sun-line";

// We validate if the user previously chose a topic
if (selectedTheme) {
  // If the validation is fulfilled, we ask what the issue was to know if we activated or deactivated the dark
  document.body.classList[selectedTheme === "dark" ? "add" : "remove"](
    darkTheme,
  );
  themeButton.classList[selectedIcon === "ri-moon-line" ? "add" : "remove"](
    iconTheme,
  );
}

// Activate / deactivate the theme manually with the button
themeButton.addEventListener("click", () => {
  // Add or remove the dark / icon theme
  document.body.classList.toggle(darkTheme);
  themeButton.classList.toggle(iconTheme);
  // We save the theme and the current icon that the user chose
  localStorage.setItem("selected-theme", getCurrentTheme());
  localStorage.setItem("selected-icon", getCurrentIcon());
});

/*=============== FILE TEXT ===============*/
$("#file").on("change", function () {
  var i = $(this).prev("label").clone();
  var file = $("#file")[0].files[0].name;
  $(this).prev("label").text(file);
});

/*=============== LOCATION SELECTOR ===============*/

$("#location").on("change", function () {
  var i = $("#location")[0].selectedIndex;
  // console.log(i)
  var location_1 = document.querySelectorAll(".location_1");
  var location_2 = document.querySelectorAll(".location_2");
  var location_3 = document.querySelectorAll(".location_3");
  if (i == 0) {
    for (var i = 0; i < location_1.length; ++i) {
      location_1[i].classList.remove("hide");
    }
    for (var i = 0; i < location_2.length; ++i) {
      location_2[i].classList.add("hide");
    }
    for (var i = 0; i < location_3.length; ++i) {
      location_3[i].classList.add("hide");
    }
  } else if (i == 1) {
    for (var i = 0; i < location_2.length; ++i) {
      location_2[i].classList.remove("hide");
    }
    for (var i = 0; i < location_1.length; ++i) {
      location_1[i].classList.add("hide");
    }
    for (var i = 0; i < location_3.length; ++i) {
      location_3[i].classList.add("hide");
    }
  } else {
    for (var i = 0; i < location_3.length; ++i) {
      location_3[i].classList.remove("hide");
    }
    for (var i = 0; i < location_2.length; ++i) {
      location_2[i].classList.add("hide");
    }
    for (var i = 0; i < location_1.length; ++i) {
      location_1[i].classList.add("hide");
    }
  }
});
