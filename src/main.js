import cues from '../assets/reference-cues.json';
import { loadCharacter } from './character.js';
import { dabWeight, DAB_DURATION } from './dab.js';
import { Performance } from './performance.js';
import { StudioSound } from './sound.js';
import { Studio } from './studio.js';

const stage = document.querySelector('#stage');
const canvas = document.querySelector('#view');
const status = document.querySelector('#status');
const audioNotice = document.querySelector('#audio-notice');
const soundButton = document.querySelector('#sound');
const resetButton = document.querySelector('#reset');
const dabButton = document.querySelector('#dab');
const soundLabel = soundButton.querySelector('span');

function readModel() {
  const element = document.querySelector('#character-data');
  const bytes = atob(element.textContent.trim());
  const data = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) data[i] = bytes.charCodeAt(i);
  return data.buffer;
}

function showError(message, error) {
  console.error(error);
  status.textContent = message;
  status.hidden = false;
  soundButton.disabled = resetButton.disabled = dabButton.disabled = true;
}

async function start() {
  let studio;
  try {
    studio = new Studio(canvas, stage);
  } catch (error) {
    showError('Для просмотра сцены открой файл в браузере с WebGL 2 и аппаратным ускорением.', error);
    return;
  }

  const character = await loadCharacter(readModel(), __EYE_MOTION__);
  studio.scene.add(character.scene);
  const [seed] = crypto.getRandomValues(new Uint32Array(1));
  const timeline = new Performance(cues, seed);
  const sound = new StudioSound(cues, timeline);
  let origin = performance.now() / 1000;
  let raf = 0;
  let running = true;
  let soundUnavailable = false;
  let dabStartedAt = null;
  const visualTime = () => Math.max(0, performance.now() / 1000 - origin);

  function frame(now) {
    if (!running) return;
    try {
      const elapsed = sound.enabled ? sound.time() : Math.max(0, now / 1000 - origin);
      const gestureTime = dabStartedAt === null ? 0 : (now - dabStartedAt) / 1000;
      if (gestureTime >= DAB_DURATION) {
        dabStartedAt = null;
        dabButton.disabled = false;
      }
      character.update(elapsed, timeline, dabStartedAt === null ? 0 : dabWeight(gestureTime));
      studio.render(now);
      raf = requestAnimationFrame(frame);
    } catch (error) {
      running = false;
      if (sound.enabled) sound.stop();
      showError('Не удалось отрисовать сцену. Попробуй перезагрузить страницу.', error);
    }
  }

  dabButton.addEventListener('click', () => {
    if (!running || dabStartedAt !== null) return;
    dabStartedAt = performance.now();
    dabButton.disabled = true;
  });
  resetButton.addEventListener('click', () => studio.reset());
  soundButton.addEventListener('click', async () => {
    if (!running) return;
    soundButton.disabled = true;
    audioNotice.hidden = true;
    try {
      if (sound.enabled) {
        origin = performance.now() / 1000 - sound.stop();
      } else {
        await sound.start(visualTime);
        if (!running) sound.stop();
      }
    } catch (error) {
      if (sound.enabled) origin = performance.now() / 1000 - sound.stop();
      console.error(error);
      soundUnavailable = !(window.AudioContext || window.webkitAudioContext);
      audioNotice.textContent = 'Не удалось включить звук. Проверь разрешение на воспроизведение аудио.';
      audioNotice.hidden = false;
    } finally {
      soundButton.dataset.playing = String(sound.enabled);
      soundLabel.textContent = sound.enabled ? 'Выключить звук' : 'Включить звук';
      soundButton.disabled = !running || soundUnavailable;
    }
  });

  canvas.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    running = false;
    cancelAnimationFrame(raf);
    if (sound.enabled) sound.stop();
    showError('Графика приостановилась. Перезагрузи страницу.', event);
  });

  status.hidden = true;
  soundButton.disabled = resetButton.disabled = dabButton.disabled = false;
  frame(origin * 1000);
}

start().catch(error => showError('Не удалось подготовить 3D-модель. Попробуй открыть index.html снова.', error));
