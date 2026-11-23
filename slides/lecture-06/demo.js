import { probabilities, seededRandom, draw } from './sampling.js';

export async function initialize() {
  const plot = document.querySelector('#sampling-plot');
  const status = document.querySelector('#sampling-status');
  const temperatureButton = document.querySelector('#temperature-button');
  const nucleusButton = document.querySelector('#nucleus-button');
  const temperatures = [1, 0.7, 1.5], nuclei = [0.6, 0.9, 1], labels = ['A', 'B', 'C', 'D'];
  const chinese = document.querySelector('.slides').lang === 'zh-CN';
  let temperatureIndex = 0, nucleusIndex = 0, random = seededRandom(29), draws = 0;
  const update = async () => {
    const temperature = temperatures[temperatureIndex], p = nuclei[nucleusIndex];
    const result = probabilities(temperature, p);
    temperatureButton.textContent = `τ = ${temperature}`;
    nucleusButton.textContent = `p = ${p}`;
    await Plotly.restyle(plot, { y: [result.before, result.after] }, [0, 1]);
    const retained = result.retained.map(i => labels[i]).join(', ');
    status.textContent = chinese
      ? `保留 ${retained}；概率质量 ${result.mass.toFixed(3)}。玩具 logits 固定；浏览器种子 29。`
      : `Retain ${retained}; mass ${result.mass.toFixed(3)}. Toy logits fixed; browser seed 29.`;
    plot.dataset.probabilities = JSON.stringify(result.after);
    return result;
  };
  temperatureButton.addEventListener('click', () => {
    temperatureIndex = (temperatureIndex + 1) % temperatures.length;
    void update();
  });
  nucleusButton.addEventListener('click', () => {
    nucleusIndex = (nucleusIndex + 1) % nuclei.length;
    void update();
  });
  document.querySelector('#draw-button').addEventListener('click', () => {
    const result = probabilities(temperatures[temperatureIndex], nuclei[nucleusIndex]);
    const token = labels[draw(result.after, random())];
    draws++;
    status.textContent = chinese
      ? `第 ${draws} 次抽取：${token}。τ = ${temperatures[temperatureIndex]}，p = ${nuclei[nucleusIndex]}；浏览器种子 29。`
      : `Draw ${draws}: ${token}. τ = ${temperatures[temperatureIndex]}, p = ${nuclei[nucleusIndex]}; browser seed 29.`;
  });
  document.querySelector('#reset-button').addEventListener('click', () => {
    temperatureIndex = 0;
    nucleusIndex = 0;
    random = seededRandom(29);
    draws = 0;
    void update();
  });
  await update();
}
