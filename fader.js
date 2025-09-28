const text1_apple = document.getElementById('text1-apple');
const text2_apple = document.getElementById('text2-apple');

function getRandomColor() {
    // Generate a random number between 0 and 255 for each color component
    const red = Math.floor(Math.random() * 256);
    const green = Math.floor(Math.random() * 256);
    const blue = Math.floor(Math.random() * 256);

    // Convert each color component to a hexadecimal string
    const redHex = red.toString(16).padStart(2, '0');
    const greenHex = green.toString(16).padStart(2, '0');
    const blueHex = blue.toString(16).padStart(2, '0');

    // Combine them into a full color string in hexadecimal format
    return `#${redHex}${greenHex}${blueHex}`;
}

function getRandomFont() {
    // Array containing the names of various fonts
    const fonts = [
        'Arial', 'Verdana', 'Helvetica', 'Times New Roman', 'Courier New',
        'Georgia', 'Palatino', 'Garamond', 'Bookman', 'Comic Sans MS',
        'Trebuchet MS', 'Arial Black', 'Impact', 'Lucida Sans Unicode'
    ];

    // Generate a random index based on the length of the fonts array
    const randomIndex = Math.floor(Math.random() * fonts.length);

    // Return the font name at the random index
    return fonts[randomIndex];
}

function getNextTitle() {
  const titles = [
    "The Platonic Representation Hypothesis",
    "L'Hypothèse de la Représentation Platonique",
    "La Hipótesis de la Representación Platónica",
    "柏拉图表征假说",
    "فرضية التمثيل الأفلاطوني",
    "प्लेटोनिक प्रतिनिधित्व परिकल्पना"
  ]

  // Return the title name at the random index
  return titles[index % titles.length];
}

function getNextAppleText() {
  const apple_texts = [
    "\"apple\"",
    "\"Apfel\"",
    "\u00A0\"سیب\""
  ]

  return apple_texts[index1 % apple_texts.length];
}

function getNextAppleEmoji() {
  const apple_emojis = [
    "\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0🍎",
    "\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0🍏"
  ]

  return apple_emojis[index2 % apple_emojis.length];
}

// Function to switch texts
index1 = 1
function switchTexts1(text) {
  text.textContent = getNextAppleText(index1);
  index1 += 1
}

index2 = 1
function switchTexts2(text) {
  text.textContent = getNextAppleEmoji(index2);
  index2 += 1
}

// Define the delay in milliseconds for the initial delay and the interval
const initialDelay1 = 3000; // 3 seconds delay before starting the interval
const intervalTime1 = 4000; // Interval time of 4 seconds
const initialDelay2 = 1000;
const intervalTime2 = 4000;

// Set a timeout to delay the start of the interval
setTimeout(() => {
    // First call to switchTexts for text1 immediately after 6 seconds
    switchTexts1(text1_apple);

    // Then set up the interval that repeats every intervalTime1 milliseconds
    setInterval(() => {
        switchTexts1(text1_apple)
    }, intervalTime1);
}, initialDelay1); // Initial 6-second delay before the first call

// Set a timeout to delay the start of the interval
setTimeout(() => {
    // First call to switchTexts for text2 immediately after 2 seconds
    switchTexts2(text2_apple);

    // Then set up the interval that repeats every intervalTime2 milliseconds
    setInterval(() => {
        switchTexts2(text2_apple)
    }, intervalTime2);
}, initialDelay2); // Initial 2-second delay before the first call
