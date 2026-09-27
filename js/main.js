// Mirror landing page: small progressive enhancements. Works fine without JS.
(function () {
  document.documentElement.classList.add("js");

  // Footer year
  var year = document.getElementById("year");
  if (year) year.textContent = new Date().getFullYear();

  // Mobile nav
  var nav = document.querySelector(".nav");
  var toggle = document.querySelector(".nav__toggle");
  if (nav && toggle) {
    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    });
    nav.querySelectorAll(".nav__links a").forEach(function (a) {
      a.addEventListener("click", function () {
        nav.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  // Reveal on scroll
  var items = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    items.forEach(function (el, i) {
      el.style.transitionDelay = (i % 4) * 70 + "ms";
      io.observe(el);
    });
  } else {
    items.forEach(function (el) { el.classList.add("is-visible"); });
  }

  // Waitlist form.
  // If the form's action points at a real endpoint (e.g. Formspree), it posts there via fetch.
  // Otherwise it just shows a local confirmation, so the static page still feels complete.
  var form = document.querySelector(".waitlist__form");
  var note = document.querySelector(".waitlist__note");
  if (form && note) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var input = form.querySelector("input[type=email]");
      var email = (input.value || "").trim();
      note.classList.remove("is-error");

      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        note.textContent = "Please enter a valid email address.";
        note.classList.add("is-error");
        input.focus();
        return;
      }

      var action = form.getAttribute("action");
      var done = function () {
        note.textContent = "You're on the list. Your Mirror will be in touch.";
        form.reset();
      };

      if (action && action !== "#") {
        fetch(action, {
          method: "POST",
          headers: { Accept: "application/json" },
          body: new FormData(form)
        }).then(function (res) {
          if (res.ok) done();
          else throw new Error();
        }).catch(function () {
          note.textContent = "Something went wrong. Please try again.";
          note.classList.add("is-error");
        });
      } else {
        done();
      }
    });
  }
})();
