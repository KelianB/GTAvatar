
// ##################################################
// Before-After Sliders
// ##################################################

function initBeforeAfterSlider(container) {
    const beforeContent = container.querySelector(".bal-before-content");
    const afterContent = container.querySelector(".bal-after-content");

    const after = document.createElement("div");
    after.className = "bal-after";

    const before = document.createElement("div");
    before.className = "bal-before";
    const beforeInset = document.createElement("div");
    beforeInset.className = "bal-before-inset";

    before.appendChild(beforeInset);

    const handle = document.createElement("div");
    handle.className = "bal-handle";
    handle.innerHTML = `
        <span class="handle-left-arrow"></span>
        <span class="handle-right-arrow"></span>
    `;

    // Move user-provided content into the correct containers
    beforeContent.remove();
    beforeInset.appendChild(beforeContent);
    afterContent.remove();
    after.appendChild(afterContent);
    // Populate the main container
    container.appendChild(after);
    container.appendChild(before);
    container.appendChild(handle);

    // Ensure beforeInset's width is always equal to the container's
    new ResizeObserver((entries) => {
        beforeInset.setAttribute("style", `width: ${entries[0].contentRect.width}px;`);
    }).observe(container);

    before.setAttribute("style", "width: 50%;");
    handle.setAttribute("style", "left: 50%;");

    const onMove = (screenX) => {
        const containerRect = container.getBoundingClientRect();
        const x = screenX - containerRect.left;
        if (x > 10 && x < containerRect.width - 10) {
            const newWidth = x * 100 / containerRect.width;
            before.setAttribute("style", "width:" + newWidth + "%;");
            handle.setAttribute("style", "left:" + newWidth + "%;");
        }

    };

    // Touch screen event listener
    container.addEventListener("touchmove", (e) => {
        onMove(e.changedTouches[0].clientX);
    });

    // Mouse move event listener
    container.addEventListener("mousemove", (e) => {
        onMove(e.clientX);
    })
}
