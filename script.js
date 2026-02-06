// ##################################################
// Theme toggling
// ##################################################

const DEFAULT_THEME = "light";

const themeToggle = document.createElement("button");
themeToggle.className = "theme-toggle";

function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    document.documentElement.setAttribute("data-bs-theme", theme == "dark" ? "light" : "dark");
    localStorage.setItem("theme", theme);
    themeToggle.innerHTML = `<span class="material-icons-outlined">${theme === "dark" ? "light_mode" : "dark_mode"}</span>`;
}

themeToggle.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    setTheme(currentTheme === "dark" ? "light" : "dark");
});

// Set default theme or retrieved saved value
const savedTheme = localStorage.getItem("theme") || DEFAULT_THEME;
setTheme(savedTheme);

document.addEventListener("DOMContentLoaded", () => {
    setTheme(savedTheme);
    document.body.appendChild(themeToggle);
});

// ##################################################
// Carousels
// ##################################################

function makeCarouselIndicators(n, carouselId) {
    // e.g.
    // <button type="button" data-bs-target="#carousel-videos" data-bs-slide-to="0" class="active" aria-current="true" aria-label="Slide 1"></button>
    // <button type="button" data-bs-target="#carousel-videos" data-bs-slide-to="1" aria-label="Slide 2"></button>

    const buttons = [];
    for (let i = 0; i < n; i++) {
        const button = document.createElement("button");
        button.setAttribute("type", "button");
        button.setAttribute("data-bs-target", carouselId);
        button.setAttribute("data-bs-slide-to", `${i}`);
        button.setAttribute("aria-label", `Slide ${i + 1}`);
        buttons.push(button);
    }
    buttons[0].className = "active";
    buttons[0].setAttribute("aria-current", "true");

    const indicators = document.createElement("div");
    indicators.className = "carousel-indicators";
    indicators.append(...buttons);
    return indicators;
}

function populateCarousel(id, nSlides, fn) {
    const carousel = document.querySelector(id);
    carousel.classList.add("carousel", "slide");

    const inner = document.createElement("div")

    inner.className = "carousel-inner";
    for (let i = 0; i < nSlides; i++) {
        inner.innerHTML += fn(i);
    }
    inner.innerHTML += `
        <button class="carousel-control-prev" type="button" data-bs-target="${id}" data-bs-slide="prev">
            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Previous</span>
        </button>
        <button class="carousel-control-next" type="button" data-bs-target="${id}" data-bs-slide="next">
            <span class="carousel-control-next-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Next</span>
        </button>
    `;

    carousel.append(makeCarouselIndicators(nSlides, id), inner);
}

document.addEventListener("DOMContentLoaded", () => {
    const RECONSTRUCTION_VIDEOS = [
        "wojtek", "person4", "marcia", "nf03"
    ];
    populateCarousel("#carousel-reconstruction-videos", RECONSTRUCTION_VIDEOS.length, (index) => {
        const name = RECONSTRUCTION_VIDEOS[index];
        return `
            <div class="carousel-item ${index === 0 ? "active" : ""}">
                <div class="carousel-content">
                    <div class="bal-container zoomable">
                        <div class="bal-before-content">
                            <video src="static/reconstruction_videos/${name}_gt.mp4" autoplay muted loop></video>
                            <span class="bal-label">Ground truth</span>
                        </div>
                        <div class="bal-after-content">
                            <video src="static/reconstruction_videos/${name}_render.mp4" autoplay muted loop></video>
                            <span class="bal-label">Render</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

    const RELIGHTING_VIDEOS = [
        ["bala_gt", "bala_red_wall"], ["malte_gt", "malte_brown_photostudio_01"], ["tom_gt", "tom_qwantani_dusk_2"], ["nf01_gt", "nf01_table_mountain_2"]
    ];
    populateCarousel("#carousel-relighting-videos", RELIGHTING_VIDEOS.length, (index) => {
        const x = RELIGHTING_VIDEOS[index];
        return `
            <div class="carousel-item ${index === 0 ? "active" : ""}">
                <div class="carousel-content">
                    <div class="bal-container zoomable">
                        <div class="bal-before-content">
                            <video src="static/relight_videos/${x[0]}.mp4" autoplay muted loop></video>
                            <span class="bal-label">Ground truth</span>
                        </div>
                        <div class="bal-after-content">
                            <video src="static/relight_videos/${x[1]}.mp4" autoplay muted loop></video>
                            <span class="bal-label">Relight</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });



    const TEXTUREEDIT_EXAMPLES = [
        ["elijah_star", "Tattoo"], ["marcel_sharp", "Checkerboard & text"], ["veronica_makeup", "Make-up"], ["obama_teaser", ""], ["katie_hair", "Hair color"], ["bala_swap", "Texture swap"], ["wojtek_swap", "Texture swap"]
    ];
    populateCarousel("#carousel-textureedit", TEXTUREEDIT_EXAMPLES.length, (index) => {
        const x = TEXTUREEDIT_EXAMPLES[index];
        return `
            <div class="carousel-item ${index === 0 ? "active" : ""}">
                <div class="carousel-content">
    
                    <div class="bal-container zoomable">
                        <div class="bal-before-content">
                            <img src="static/tex_edit/${x[0]}/reconstruction.png" alt="Render" />
                            <span class="bal-label">Render</span>
                        </div>
                        <div class="bal-after-content">
                            <img src="static/tex_edit/${x[0]}/edit.png" alt="Edited render" />
                            <span class="bal-label">Edit</span>
                        </div>
                    </div>
    
                    <div class="texture-frames-container small-onlyy">
                        <div class="texture-frame">
                            <img src="static/tex_edit/${x[0]}/tex_albedo.png" alt="Original texture" />
                            <span>Original</span>
                        </div>
                        <div class="texture-frame">
                            <img src="static/tex_edit/${x[0]}/tex_albedo_edit.png" alt="Edited texture" />
                            <span>Edited</span>
                        </div>
                    </div>
                </div>
                <div class="carousel-caption">
                    <h5>${x[1]}</h5>
                </div>
            </div>
        `;
    });

    const PBREDIT_EXAMPLES = [
        "bala_scales", "obama_metal", "malte_mud"
    ];
    populateCarousel("#carousel-texturepbr", PBREDIT_EXAMPLES.length, (index) => {
        const name = PBREDIT_EXAMPLES[index];
        return `
            <div class="carousel-item ${index === 0 ? "active" : ""}" data-mode="render">
                <div class="carousel-content">
                    <div class="bal-container zoomable">
                        <div class="bal-before-content">
                            <img class='is_render' src="static/tex_edit_pbr/${name}/reconstruct_render.png" alt="Render" />
                            <img class='is_normal' src="static/tex_edit_pbr/${name}/reconstruct_normals.png" alt="Normals" />
                            <span class="bal-label">Render</span>
                        </div>
                        <div class="bal-after-content">
                            <img class='is_render' src="static/tex_edit_pbr/${name}/edit_render.png" alt="Edited render" />
                            <img class='is_normal' src="static/tex_edit_pbr/${name}/edit_normals.png" alt="Edited normals" />
                            <span class="bal-label">Edit</span>
                        </div>
                    </div>

                    <div class="textures">
                        <div class="texture-frame">
                            <img src="static/tex_edit_pbr/${name}/albedo.png" alt="PBR textures" />
                            <span>Albedo</span>
                        </div>
                          <div class="texture-frame">
                            <img src="static/tex_edit_pbr/${name}/normal.png" alt="PBR textures" />
                            <span>Normal</span>
                        </div>
                          <div class="texture-frame">
                            <img src="static/tex_edit_pbr/${name}/roughness.png" alt="PBR textures" />
                            <span>Roughness</span>
                        </div>
                    </div>

                    <div class="carousel-caption">
                        <span class="normal-toggle">Normal mode&nbsp;<input type="checkbox" /></span>
                    </div>
                </div>
            </div>
        `;
        // <div class="texture-frame">
        //     <img src="static/tex_edit_pbr/${x[0]}/spec.png" alt="PBR textures" />
        //     <span>Spec. refl.</span>
        // </div>
    });

    // Toggle normal maps on and off in the PBR examples
    document.querySelectorAll(".normal-toggle input").forEach(toggle => {
        toggle.addEventListener("change", (e) => {
            const carouselItem = e.target.closest(".carousel-item");
            if (carouselItem) {
                carouselItem.setAttribute("data-mode", e.target.checked ? "normal" : "render");
            }
        });
    });
});

// ##################################################
// Misc.
// ##################################################

function copyToClipboard(elementSelector) {
    const element = document.querySelector(elementSelector);
    if (!element) return;
    navigator.clipboard.writeText(element.textContent);
}

document.addEventListener("DOMContentLoaded", () => {
    [...document.getElementsByClassName("bal-container")].forEach(initBeforeAfterSlider);
    document.getElementById("copy-bibtex").addEventListener("click", () => copyToClipboard("#bibtex code"));
    makeZoomable(".zoomable", 2);
});
