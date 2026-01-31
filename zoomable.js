
// ##################################################
// Zoomable elements
// ##################################################

const is = (el, className) => el.classList.contains(className);

function makeZoomable(selector, zoomScale) {
    let elements = document.querySelectorAll(selector);

    let zoomedElement = null;
    let elMouseDownX = 0, elMouseDownY = 0;
    let tx = 0, ty = 0;

    elements.forEach((element) => {
        // Zoom on click
        element.addEventListener("click", (e) => {
            if (!zoomedElement && !is(element, "zoomed")) {
                zoomedElement = element;
                element.classList.add("zoomed");
                element.style.scale = zoomScale;
                document.body.style.cursor = "zoom-out";
                e.preventDefault();
                e.stopPropagation();
            }
        });

        // Initiate dragging on mousedown
        element.addEventListener("mousedown", (e) => {
            if (is(element, "zoomed") && !is(element, "dragging")) {
                element.classList.add("dragging");

                const currentTranslate = element.style.translate || "0px 0px";
                let match = currentTranslate.match(/(-?\d+\.?\d+)px (-?\d+\.?\d+)px/);
                if (match) {
                    tx = parseInt(match[1]);
                    ty = parseInt(match[2]);
                } else {
                    match = currentTranslate.match(/(-?\d+\.?\d+)px/);
                    if (match) {
                        tx = ty = parseInt(match[1]);
                    }
                }

                elMouseDownX = e.clientX;
                elMouseDownY = e.clientY;
            }
        });

        // Handle drag movement
        element.addEventListener("mousemove", (e) => {
            if (is(element, "dragging")) {
                const newX = tx + e.clientX - elMouseDownX;
                const newY = ty + e.clientY - elMouseDownY;
                element.style.translate = `${newX}px ${newY}px`;
            }
        });
        // Handle zooming with mouse wheel
        element.addEventListener("wheel", (e) => {
            if (is(element, "zoomed")) {
                const delta = e.deltaY || e.detail || e.wheelDelta;
                const currentScale = parseFloat(element.style.scale || 1);
                element.style.scale = Math.max(1, currentScale + (delta > 0 ? -0.2 : 0.2));
                e.preventDefault();
                e.stopPropagation();
            }
        });
    });

    // Keep track of mouse down position on the document to distinguish between click and drag when releasing
    let documentMouseDownX = 0, documentMouseDownY = 0;
    document.addEventListener("mousedown", (e) => {
        documentMouseDownX = e.clientX;
        documentMouseDownY = e.clientY;
    });
    // Handle drag release and unzoom
    document.addEventListener("mouseup", (e) => {
        if (zoomedElement) {
            if (is(zoomedElement, "dragging")) {
                zoomedElement.classList.remove("dragging");
            }
            // Check if the user has dragged or just clicked
            if (Math.abs(e.clientX - documentMouseDownX) < 5 && Math.abs(e.clientY - documentMouseDownY) < 5) {
                zoomedElement.classList.remove("zoomed");
                zoomedElement.style.scale = 1;
                zoomedElement.style.translate = "0px 0px";
                document.body.style.cursor = "initial";
                elMouseDownX = elMouseDownY = tx = ty = 0;

                // Delay clearing zoomedElement to allow click event to propagate and not immediately re-zoom
                setTimeout(() => (zoomedElement = null), 100);
            }
        }
    });
}
