importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow-models/transformers@0.0.4/dist/transformers.min.js');

let model, tokenizer;

async function init() {
  try {
    const modelKey = 'distilbert-model';
    model = await tf.io.browser.fromIndexedDB(modelKey);
    if (!model) {
      model = await transformers.loadModel('distilbert', 'distilbert-base-uncased-finetuned-sst-2-english');
      await model.save(`indexeddb://${modelKey}`);
    }
    tokenizer = await transformers.loadTokenizer('distilbert', 'distilbert-base-uncased-finetuned-sst-2-english');
  } catch (error) {
    self.postMessage({ error: error.message });
  }
}

function generateHumanLikeResponse(output, inputText, isFirstInteraction, username) {
  const sentiment = output.label === 'POSITIVE' ? 'positive' : output.label === 'NEGATIVE' ? 'negative' : 'neutral';
  const confidence = output.score;
  const words = inputText.toLowerCase().split(' ').filter(w => w.length > 3);
  const keyword = words.find(w => ['happy', 'sad', 'great', 'bad', 'okay'].includes(w)) || sentiment;

  const templates = {
    positive: [
      `Wow, you're full of ${confidence > 0.8 ? 'amazing' : 'great'} ${keyword} vibes! What's the story?`,
      `That’s super uplifting! What’s sparking this ${keyword} mood?`,
      `Loving this ${keyword} energy! Tell me more!`,
      `You're radiating ${keyword}! What's got you so pumped?`
    ],
    neutral: [
      `Sounds like you're keeping it ${keyword}. What's up?`,
      `Pretty chill ${keyword} vibe! What's on your mind?`,
      `Just cruising with ${keyword}? Got anything exciting to share?`,
      `Keeping it ${keyword}, huh? What's next?`
    ],
    negative: [
      `Sorry you're feeling ${keyword}. Want to talk about it?`,
      `That sounds tough with ${keyword}. I'm here if you need to vent!`,
      `Ouch, a ${keyword} moment? Let’s chat about it.`,
      `Feeling ${keyword}? I'm all ears if you want to share.`
    ]
  };

  const selectedTemplates = templates[sentiment];
  let response = selectedTemplates[Math.floor(Math.random() * selectedTemplates.length)];
  if (inputText.includes('?')) {
    response += ' By the way, what’s the answer to your question? I’m curious!';
  }
  if (isFirstInteraction && username) {
    response = `Hey ${username}, welcome to Vynix AI! ${response}`;
  }
  return response;
}

self.onmessage = async (e) => {
  const { input, isFirstInteraction, username } = e.data;
  try {
    if (!model) await init();
    const tokens = await tokenizer.encode(input);
    if (!tokens) throw new Error('Invalid input');
    const tensor = tf.tensor([tokens]);
    const prediction = await model.predict(tensor);
    const output = { label: prediction.label, score: prediction.score };
    const response = generateHumanLikeResponse(output, input, isFirstInteraction, username);
    self.postMessage({ response });
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
