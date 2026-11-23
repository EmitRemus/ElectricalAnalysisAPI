const filter = document.querySelector("[data-route-filter]");

if (filter) {
  const groups = [...document.querySelectorAll("[data-route-group]")];
  const navLinks = [...document.querySelectorAll(".toc a")];
  const status = document.querySelector("[data-filter-status]");
  const entries = groups.map(group => ({
    group,
    link: navLinks.find(link => link.hash === "#" + group.id),
    routes: [...group.querySelectorAll("[data-route]")].map(route => ({
      route,
      label: route.querySelector("summary").textContent.toLocaleLowerCase("uk"),
    })),
  }));

  filter.closest(".filter").hidden = false;
  filter.addEventListener("input", () => {
    const query = filter.value.toLocaleLowerCase("uk").trim();
    let count = 0;
    for (const { group, link, routes } of entries) {
      for (const { route, label } of routes) {
        route.hidden = !label.includes(query);
        if (!route.hidden) count += 1;
      }
      group.hidden = routes.every(({ route }) => route.hidden);
      if (link) link.hidden = group.hidden;
    }
    status.textContent = query ? (count ? "Знайдено маршрутів: " + count : "Маршрутів не знайдено") : "";
  });
}

function openLinkedRoute() {
  const target = document.getElementById(location.hash.slice(1));
  if (target?.matches("details[data-route]")) target.open = true;
}
openLinkedRoute();
window.addEventListener("hashchange", openLinkedRoute);

for (const button of document.querySelectorAll("[data-copy]")) {
  const code = button.closest(".sample").querySelector("pre code");
  const originalLabel = button.textContent;
  let restoreLabel;
  button.hidden = false;
  button.addEventListener("click", async () => {
    clearTimeout(restoreLabel);
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = "Скопійовано";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Виділено · Ctrl+C";
    }
    restoreLabel = setTimeout(() => { button.textContent = originalLabel; }, 2500);
  });
}
