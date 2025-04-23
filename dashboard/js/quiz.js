// ========== QUESTIONS SETUP ==========
const questions = {
    OPN: [
        { prompt: "I have a vivid imagination.", reverse: false },
        { prompt: "I have excellent ideas.", reverse: false },
        { prompt: "I am full of ideas.", reverse: false },
        { prompt: "I am quick to understand things.", reverse: false },
        { prompt: "I use difficult words.", reverse: false },
        { prompt: "I spend time reflecting on things.", reverse: false },
        { prompt: "I am interested in abstract ideas.", reverse: false },
        { prompt: "I do not have a good imagination.", reverse: true },
        { prompt: "I have difficulty understanding abstract ideas.", reverse: true },
        { prompt: "I do not enjoy thinking about theoretical ideas.", reverse: true }
    ],
    CSN: [
        { prompt: "I am always prepared.", reverse: false },
        { prompt: "I follow a schedule.", reverse: false },
        { prompt: "I get chores done right away.", reverse: false },
        { prompt: "I pay attention to details.", reverse: false },
        { prompt: "I like order.", reverse: false },
        { prompt: "I make plans and stick to them.", reverse: false },
        { prompt: "I do things according to a plan.", reverse: false },
        { prompt: "I waste my time.", reverse: true },
        { prompt: "I do not finish what I start.", reverse: true },
        { prompt: "I find it difficult to get down to work.", reverse: true }
    ],
    EXT: [
        { prompt: "I am the life of the party.", reverse: false },
        { prompt: "I talk a lot.", reverse: false },
        { prompt: "I keep in the background.", reverse: true },
        { prompt: "I do not talk a lot.", reverse: true },
        { prompt: "I feel comfortable around people.", reverse: false },
        { prompt: "I start conversations.", reverse: false },
        { prompt: "I have little to say.", reverse: true },
        { prompt: "I don’t mind being the center of attention.", reverse: false },
        { prompt: "I am quiet around strangers.", reverse: true },
        { prompt: "I am reserved.", reverse: true }
    ],
    AGR: [
        { prompt: "I am interested in people.", reverse: false },
        { prompt: "I sympathize with others’ feelings.", reverse: false },
        { prompt: "I have a soft heart.", reverse: false },
        { prompt: "I take time out for others.", reverse: false },
        { prompt: "I make people feel at ease.", reverse: false },
        { prompt: "I feel others' emotions.", reverse: false },
        { prompt: "I have difficulty understanding others.", reverse: true },
        { prompt: "I am not really interested in others.", reverse: true },
        { prompt: "I insult people.", reverse: true },
        { prompt: "I feel little concern for others.", reverse: true }
    ],
    EST: [
        { prompt: "I get stressed out easily.", reverse: false },
        { prompt: "I worry a lot.", reverse: false },
        { prompt: "I am easily disturbed.", reverse: false },
        { prompt: "I get upset easily.", reverse: false },
        { prompt: "I have frequent mood swings.", reverse: false },
        { prompt: "I get irritated easily.", reverse: false },
        { prompt: "I often feel blue.", reverse: false },
        { prompt: "I am relaxed most of the time.", reverse: true },
        { prompt: "I seldom feel blue.", reverse: true },
        { prompt: "I do not get stressed out easily.", reverse: true }
    ]
};

const responses = [
  { label: 'Strongly Agree', value: 5 },
  { label: 'Agree', value: 4 },
  { label: 'Neutral', value: 3 },
  { label: 'Disagree', value: 2 },
  { label: 'Strongly Disagree', value: 1 }
];


// ========== LOGIC ==========
const scores = { OPN: 0, CSN: 0, EXT: 0, AGR: 0, EST: 0 };
let answered = 0;
let total = 50;
let startTime = null;

// DOM READY
$(document).ready(function () {
    buildQuiz();
    $('#submit-btn').click(handleSubmit);
    $('#retake-btn').click(() => location.reload());
    $('#start-over-btn').click(() => location.reload());
    startTime = new Date();
});

function shuffle(array) {
    return array.sort(() => Math.random() - 0.5);
}

function buildQuiz() {
    const quiz = $('#quiz');
    const questionPool = [];

    // Flatten and tag questions
    for (const trait in questions) {
        questions[trait].forEach((q, i) => {
            questionPool.push({
                trait: trait,
                prompt: q.prompt,
                reverse: q.reverse,
                class: `${trait}${i}`
            });
        });
    }

    // Shuffle all questions
    const shuffled = shuffle(questionPool);
    total = shuffled.length;

    // Render questions
    shuffled.forEach((q, index) => {
        const html = `
            <li class="prompt">
                <div>${q.prompt}</div>
                <div class="btn-group">
                    ${responses.map(r => `
                        <button class="btn value-btn ${q.class}"
                                data-trait="${q.trait}"
                                data-reverse="${q.reverse}"
                                data-value="${r.value}">
                            ${r.label}
                        </button>
                    `).join('')}
                </div>
            </li>
        `;
        quiz.append(html);
    });

    // event listeners after DOM elements exist
    $('.value-btn').click(handleAnswer);
}

function handleAnswer() {
    const btn = $(this);
    const trait = btn.data('trait');
    const reverse = btn.data('reverse') === true || btn.data('reverse') === "true";
    const value = parseInt(btn.data('value'));
    const questionClass = btn.attr('class').split(" ")[2];

    const group = $(`.${questionClass}`);
    const old = group.filter('.active');

    if (!old.length) answered++;
    else {
        const oldVal = parseInt(old.data('value'));
        const wasReversed = old.data('reverse') === true || old.data('reverse') === "true";
        scores[trait] -= wasReversed ? reverseScore(oldVal) : oldVal;
    }

    group.removeClass('active');
    btn.addClass('active');

    const finalVal = reverse ? reverseScore(value) : value;
    scores[trait] += finalVal;

    updateProgressBar();
    $('#submit-btn').prop('disabled', answered < total);
    console.log(`Answered ${answered} / ${total}`);
}


function handleSubmit() {
    const timeSpent = Math.floor((new Date() - startTime) / 1000);

    // Build the full list of 50 raw responses
    let rawResponses = [];

    for (const trait of ["OPN", "CSN", "EXT", "AGR", "EST"]) {
        for (let i = 0; i < 10; i++) {
            const questionClass = `${trait}${i}`;
            const activeBtn = $(`.${questionClass}.active`);
            if (activeBtn.length === 0) {
                alert("Please answer all questions.");
                return;
            }

            const value = parseInt(activeBtn.data("value"));
            const reverse = activeBtn.data("reverse") === true || activeBtn.data("reverse") === "true";
            const finalVal = reverse ? reverseScore(value) : value;

            rawResponses.push(finalVal);
        }
    }

    // Save scores in localStorage
    const data = {
        scores: scores,
        features: rawResponses,
        timeSpent: timeSpent
    };

    localStorage.setItem('personalityTestData', JSON.stringify(data));

    fetch('http://localhost:5000/predict-cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: rawResponses })
    })
    .then(res => res.json())
    .then(response => {
        console.log("Predicted cluster:", response.cluster);
        showResults(scores, response.cluster, timeSpent);
    })
    .catch(err => {
        console.error("Error predicting cluster:", err);
        showResults(scores, "Unavailable", timeSpent);
    });

    $('#quiz').hide();
    $('#submit-btn').hide();
    $('#retake-btn').show();
    $('.results').removeClass('hide');
}

function reverseScore(value) {
  return 6 - value;
}


function updateProgressBar() {
    const percent = (answered / total) * 100;
    $('#progress').css('width', `${percent}%`);
}

function handleSubmit() {
    const timeSpent = Math.floor((new Date() - startTime) / 1000);
    const data = { scores, timeSpent };

    // Save locally
    localStorage.setItem('personalityTestData', JSON.stringify(data));

    fetch('http://localhost:5000/predict-cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: Object.values(scores) })
    })
    .then(res => res.json())
    .then(response => {
        console.log("Predicted cluster:", response.cluster);
        showResults(scores, response.cluster, timeSpent);
    })
    .catch(err => {
        console.error("Error predicting cluster:", err);
        showResults(scores, "Unavailable", timeSpent);
    });

    $('#quiz').hide();
    $('#submit-btn').hide();
    $('#retake-btn').show();
    $('.results').removeClass('hide');
    console.log("Scores:", scores);

}

function showResults(scores, cluster, timeSpent) {
    let html = `<h3>Your Scores</h3>`;
    for (const [trait, score] of Object.entries(scores)) {
        html += `<p><strong>${trait}</strong>: ${score}</p>`;
    }

    // Cluster Descriptions
    const clusterDescriptions = {
        0: {
            title: "Cluster 0: The Easygoing Idealist",
            desc: "You're organized, thoughtful, and emotionally stable. You enjoy structure and value depth over spontaneity.You're intellectually curious and socially cooperative, with decent emotional stability. This combination makes you a flexible team player who can navigate both new ideas and interpersonal dynamics effectively."
        },
        1: {
            title: "Cluster 1:The Content Loner",
            desc: "You’re imaginative and curious. You march to the beat of your own drum, exploring ideas and concepts with a laid-back attitude and less concern for social conventions."
        },
        2: {
            title: "The Friendly Intellectual",
            desc: "You’re curious and open-minded, yet grounded. You may be introverted but emotionally secure and thoughtful."
        },
        3: {
            title: "Cluster 3:The People Pleaser",
            desc: "Practical and emotionally steady, you approach life with measured confidence. You may not seek the spotlight, but you shine in your own way."
        }
    };

    const clusterNum = parseInt(cluster);
    const clusterInfo = clusterDescriptions[clusterNum] || {
        title: "Unknown Cluster",
        desc: "We couldn’t confidently assign you to a cluster. Try retaking the quiz!"
    };

    html += `<h3>Cluster Match: Cluster ${clusterNum + 1}</h3>`;
    html += `<p><strong>Personality Type: ${clusterInfo.title}</strong></p>`;
    html += `<p>${clusterInfo.desc}</p>`;
    html += `<p><small>Completed in ${timeSpent} seconds</small></p>`;

    $('#results').html(html);
}
