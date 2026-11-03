(function () {
    "use strict";

    const GROUPS = {
        images: {
            color: "#3c7a5c",
            desc: "<strong>Images</strong> from COCO val2017, decomposed by GRACE on top of <strong>CLIP ViT-B/32</strong>, trained on ImageNet only with a vocabulary of 15,162 lemmas mined from ImageNet captions. Retrieval on COCO is zero-shot.",
        },
        egocentric: {
            color: "#3b6fd8",
            desc: "<strong>Egocentric clips</strong> from Ego4D (with their narrations), decomposed by GRACE on top of the <strong>Perception Encoder</strong>, trained on EgoClip with a vocabulary of 700 verbs and nouns mined from its narrations. Each narrated clip lasts about a second: the videos show it within 6 s of context (at 5 fps), with the clip marked on the timeline.",
        },
        // ConVisBench, hidden (also uncomment its tab in index.html):
        // convisbench: {
            // color: "#c27a1a",
            // desc: "<strong>Out-of-domain videos</strong> from ConVisBench, decomposed by the EgoClip-trained <strong>Perception Encoder</strong> GRACE model with the full 2,139-concept vocabulary, the one used to estimate concept-based video similarity. Captions by mPLUG-Owl3. The decomposition describes the whole video, so longer videos are sped up to loop in about 12 s.",
        // },
    };

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    // ---------- Videos: muted loops that play while on screen ----------

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const visibility = "IntersectionObserver" in window ? new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            const video = entry.target;
            if (entry.isIntersecting && !reduceMotion && !video.dataset.userPaused) {
                video.play().catch(function () {});
            } else {
                video.pause();
            }
        });
    }, { threshold: 0.4 }) : null;

    // The teaser loops on its own, except with reduced motion: then it waits for the viewer
    const teaser = document.querySelector(".teaser-video");
    if (teaser && reduceMotion) {
        teaser.removeAttribute("autoplay");
        teaser.pause();
        teaser.controls = true;
    }

    function renderVideo(example, color) {
        const media = el("div", "example-media video" + (example.video.clip ? " ego" : ""));
        const video = el("video");
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = "none";
        video.poster = example.video.poster;
        video.src = example.video.src;
        video.setAttribute("aria-label", "Video: " + example.caption);
        media.appendChild(video);

        if (example.video.speed > 1) {
            media.appendChild(el("span", "video-badge", example.video.speed + "\u00d7 speed"));
        }
        // Click to pause / resume; a paused video stays paused while scrolling
        media.addEventListener("click", function () {
            if (video.paused) {
                delete video.dataset.userPaused;
                video.play().catch(function () {});
            } else {
                video.dataset.userPaused = "1";
                video.pause();
            }
        });

        if (example.video.clip) {
            // Ego4D: the decomposed clip is a short segment of the video, marked on a timeline
            const timeline = el("div", "clip-timeline");
            const segment = el("span", "clip-segment");
            segment.style.background = color;
            const playhead = el("span", "clip-playhead");
            timeline.appendChild(segment);
            timeline.appendChild(playhead);
            media.appendChild(timeline);

            const clip = example.video.clip;
            function layout() {
                const duration = video.duration || 6;
                segment.style.left = (100 * clip[0] / duration) + "%";
                segment.style.width = Math.max(1.5, 100 * (clip[1] - clip[0]) / duration) + "%";
            }
            function update() {
                const t = video.currentTime;
                playhead.style.left = (100 * t / (video.duration || 6)) + "%";
                if (!video.paused) requestAnimationFrame(update);
            }
            layout();
            video.addEventListener("loadedmetadata", layout);
            video.addEventListener("play", function () { requestAnimationFrame(update); });
            video.addEventListener("seeked", update);
        }

        if (visibility) visibility.observe(video);
        return media;
    }

    function renderExample(example, color) {
        const card = el("article", "example");

        if (example.video) {
            card.appendChild(renderVideo(example, color));
        } else {
            const media = el("div", "example-media");
            media.classList.add(example.media.length === 1 ? "single" : "frames-" + example.media.length);
            example.media.forEach(function (src, i) {
                const img = el("img");
                img.src = src;
                img.loading = "lazy";
                img.alt = example.media.length === 1 ? example.caption : "Frame " + (i + 1) + ": " + example.caption;
                media.appendChild(img);
            });
            card.appendChild(media);
        }

        const body = el("div", "example-body");
        const caption = el("p", "example-caption", "“" + example.caption + "”");
        caption.title = example.caption;
        body.appendChild(caption);
        body.appendChild(el("p", "example-source", example.source));

        const list = el("ol", "concepts");
        list.setAttribute("aria-label", "Top concepts and their reconstruction weights");
        const max = example.concepts[0][1];
        example.concepts.forEach(function (concept, i) {
            const row = el("li", "concept");
            row.appendChild(el("span", "concept-name", concept[0]));
            const bar = el("span", "concept-bar");
            const fill = el("span", "concept-fill");
            fill.style.display = "block";
            fill.style.width = (100 * concept[1] / max).toFixed(1) + "%";
            fill.style.background = color;
            fill.style.animationDelay = (i * 60) + "ms";
            bar.appendChild(fill);
            row.appendChild(bar);
            row.appendChild(el("span", "concept-weight", concept[1].toFixed(3)));
            list.appendChild(row);
        });
        body.appendChild(list);
        card.appendChild(body);
        return card;
    }

    // Optional text slots: the page may not have them
    function setText(id, text) {
        const node = document.getElementById(id);
        if (node) node.textContent = text;
    }

    // How each group's examples were chosen (`meta` in examples.js, written by docs/build_examples.py)
    function selectionNote(name) {
        const meta = ((window.GRACE_EXAMPLES || {}).meta || {})[name] || {};
        return meta.selection === "random"
            ? "Uniform random samples (seed " + meta.seed + "), not filtered."
            : "Examples hand-picked from random samples.";
    }

    function showGroup(name) {
        const data = window.GRACE_EXAMPLES || {};
        setText("examples-selection", selectionNote(name));
        const grid = document.getElementById("example-grid");
        const group = GROUPS[name];
        document.getElementById("group-desc").innerHTML = group.desc;
        grid.querySelectorAll("video").forEach(function (video) {
            video.pause();
            if (visibility) visibility.unobserve(video);
        });
        grid.replaceChildren();
        (data[name] || []).forEach(function (example) {
            grid.appendChild(renderExample(example, group.color));
        });
        document.querySelectorAll(".tab").forEach(function (tab) {
            tab.setAttribute("aria-selected", String(tab.dataset.group === name));
        });
    }

    document.querySelectorAll(".tab").forEach(function (tab) {
        tab.addEventListener("click", function () {
            showGroup(tab.dataset.group);
        });
    });
    // Deep links to a tab: index.html#egocentric, index.html#convisbench
    const linked = window.location.hash.slice(1);
    showGroup(GROUPS[linked] ? linked : "egocentric");
    requestAnimationFrame(function () { document.querySelector(".tabs").classList.add("ready"); });
    if (GROUPS[linked]) document.getElementById("examples").scrollIntoView();

    // ---------- Visual grounding: one card per ADE20k image, a button per concept swaps in its heatmap ----------

    function renderGrounding(example) {
        const card = el("article", "example grounding-card");
        const img = el("img", "grounding-img");
        img.loading = "lazy";
        card.appendChild(img);

        const body = el("div", "example-body");
        const buttons = el("div", "grounding-buttons");
        const options = example.concepts.map(function (concept, i) {
            return [concept[0], concept[1], example.overlays[i]];
        });
        // Preload the heatmaps so switching is instant
        example.overlays.forEach(function (src) { new Image().src = src; });

        function select(i) {
            img.src = options[i][2];
            img.alt = "Heatmap of the concept “" + options[i][0] + "”";
            buttons.querySelectorAll("button").forEach(function (b, j) {
                b.setAttribute("aria-pressed", String(i === j));
            });
        }
        options.forEach(function (option, i) {
            const button = el("button", "grounding-btn");
            button.type = "button";
            button.appendChild(el("span", "", option[0]));
            button.appendChild(el("span", "grounding-weight", option[1].toFixed(3)));
            button.addEventListener("click", function () { select(i); });
            button.addEventListener("mouseenter", function () { select(i); });
            button.addEventListener("focus", function () { select(i); });
            buttons.appendChild(button);
        });
        body.appendChild(buttons);
        body.appendChild(el("p", "example-source", example.source));
        card.appendChild(body);
        select(example.default || 0);
        return card;
    }

    setText("grounding-selection", selectionNote("grounding"));
    const groundingGrid = document.getElementById("grounding-grid");
    ((window.GRACE_EXAMPLES || {}).grounding || []).forEach(function (example) {
        groundingGrid.appendChild(renderGrounding(example));
    });

    // ---------- ImageNet folds (Figure 4): GRACE per fold, both methods on average ----------

    const FOLDS = [
        // [label, method, trained on all classes (w/o shift), test classes held out (w/ shift)]
        ["Fold 1", "GRACE", 49.50, 43.43],
        ["Fold 2", "GRACE", 39.10, 27.86],
        ["Fold 3", "GRACE", 54.65, 51.03],
        ["Fold 4", "GRACE", 44.59, 39.79],
        ["Fold 5", "GRACE", 48.97, 44.83],
        ["Avg.", "EXPLAIN-R", 48.15, 38.92],
        ["Avg.", "GRACE", 47.36, 41.39],
    ];
    const Y_MAX = 60;

    function renderFolds() {
        const chart = document.getElementById("fold-chart");
        const plot = el("div", "fold-plot");
        [0, 20, 40, 60].forEach(function (y) {
            const line = el("div", "fold-grid");
            line.style.bottom = (100 * y / Y_MAX) + "%";
            line.appendChild(el("span", "", String(y)));
            plot.appendChild(line);
        });
        const tooltip = el("div", "fold-tooltip");
        tooltip.hidden = true;

        const groups = {};
        FOLDS.forEach(function (row) {
            if (!groups[row[0]]) {
                groups[row[0]] = el("div", "fold-group" + (row[0] === "Avg." ? " avg" : ""));
                groups[row[0]].appendChild(el("div", "fold-bars"));
                groups[row[0]].appendChild(el("div", "fold-label", row[0]));
                plot.appendChild(groups[row[0]]);
            }
            [[row[2], false], [row[3], true]].forEach(function (bar) {
                const node = el("div", "fold-bar " + (row[1] === "GRACE" ? "grace" : "explainr") + (bar[1] ? " hatched" : ""));
                node.style.height = (100 * bar[0] / Y_MAX) + "%";
                node.tabIndex = 0;
                const text = row[1] + " · " + row[0] + " · " + (bar[1] ? "test classes held out" : "trained on all classes") + ": " + bar[0].toFixed(2) + "%";
                node.setAttribute("aria-label", text);
                if (row[0] === "Avg.") node.appendChild(el("span", "fold-value", bar[0].toFixed(2)));
                function show() {
                    tooltip.textContent = text;
                    tooltip.hidden = false;
                    const box = node.getBoundingClientRect();
                    const parent = chart.getBoundingClientRect();
                    tooltip.style.left = (box.left + box.width / 2 - parent.left) + "px";
                    tooltip.style.top = (box.top - parent.top) + "px";
                }
                node.addEventListener("mouseenter", show);
                node.addEventListener("focus", show);
                node.addEventListener("mouseleave", function () { tooltip.hidden = true; });
                node.addEventListener("blur", function () { tooltip.hidden = true; });
                groups[row[0]].firstChild.appendChild(node);
            });
        });
        chart.appendChild(el("div", "fold-ylabel", "Top-1 accuracy (%)"));
        chart.appendChild(plot);
        chart.appendChild(tooltip);
    }
    renderFolds();

    const copy = document.getElementById("copy-bibtex");
    copy.addEventListener("click", function () {
        const text = document.getElementById("bibtex").textContent;
        navigator.clipboard.writeText(text).then(function () {
            copy.textContent = "Copied!";
            setTimeout(function () { copy.textContent = "Copy"; }, 1500);
        });
    });
})();
