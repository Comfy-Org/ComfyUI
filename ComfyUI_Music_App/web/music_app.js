import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const extensionName = "ComfyUI.MusicApp";
const outputNodeId = "20";
const maximumWait = 30 * 60 * 1000;

const optionList = (node, input) => {
  const value = node?.input?.required?.[input] ?? node?.input?.optional?.[input];
  return Array.isArray(value?.[0]) ? value[0] : [];
};

function field(parent, labelText, control) {
  const label = document.createElement("label");
  label.className = "comfy-music-field";
  const caption = document.createElement("span");
  caption.textContent = labelText;
  label.append(caption, control);
  parent.append(label);
  return control;
}

function textInput(value, type = "text") {
  const input = document.createElement("input");
  input.type = type;
  input.value = value;
  return input;
}

function selectInput(options, preferred) {
  const select = document.createElement("select");
  for (const value of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
  if (options.includes(preferred)) select.value = preferred;
  return select;
}

function generatedLyrics(theme) {
  const subject = theme.trim().replace(/\s+/g, " ").slice(0, 160);
  return `[Verso 1]\nNa estrada que me leva, ${subject} vem me encontrar\nCada passo guarda um sonho, cada sonho quer cantar\n\n[Refrão]\n${subject}, fica perto de mim\nComo a luz que atravessa a noite e anuncia o jardim\n${subject}, minha voz vai seguir\nNo compasso dessa história que ainda está por vir\n\n[Verso 2]\nSe o caminho muda o rumo, deixo o coração guiar\nLevo a força da lembrança e a coragem de tentar\n\n[Refrão]\n${subject}, fica perto de mim\nComo a luz que atravessa a noite e anuncia o jardim\n${subject}, minha voz vai seguir\nNo compasso dessa história que ainda está por vir`;
}

function createWorkflow(settings) {
  return {
    "1": {
      class_type: "UNETLoader",
      inputs: { unet_name: settings.model, weight_dtype: "default" },
    },
    "2": {
      class_type: "DualCLIPLoader",
      inputs: {
        clip_name1: settings.clip1,
        clip_name2: settings.clip2,
        type: "ace",
        device: "default",
      },
    },
    "3": {
      class_type: "VAELoader",
      inputs: { vae_name: settings.vae },
    },
    "4": {
      class_type: "TextEncodeAceStepAudio1.5",
      inputs: {
        clip: ["2", 0],
        tags: settings.tags,
        lyrics: settings.lyrics,
        seed: settings.seed,
        bpm: settings.bpm,
        duration: settings.duration,
        timesignature: settings.timesignature,
        language: settings.language,
        keyscale: settings.keyscale,
        generate_audio_codes: true,
        cfg_scale: 2,
        temperature: 0.85,
        top_p: 0.9,
        top_k: 0,
        min_p: 0,
      },
    },
    "5": {
      class_type: "ConditioningZeroOut",
      inputs: { conditioning: ["4", 0] },
    },
    "6": {
      class_type: "EmptyAceStep1.5LatentAudio",
      inputs: { seconds: settings.duration, batch_size: 1 },
    },
    "7": {
      class_type: "ModelSamplingAuraFlow",
      inputs: { model: ["1", 0], shift: 3 },
    },
    "8": {
      class_type: "KSampler",
      inputs: {
        model: ["7", 0],
        positive: ["4", 0],
        negative: ["5", 0],
        latent_image: ["6", 0],
        seed: settings.seed,
        steps: 8,
        cfg: 1,
        sampler_name: "euler",
        scheduler: "simple",
        denoise: 1,
      },
    },
    "9": {
      class_type: "VAEDecodeAudio",
      inputs: { samples: ["8", 0], vae: ["3", 0] },
    },
    [outputNodeId]: {
      class_type: "SaveAudioAdvanced",
      inputs: {
        audio: ["9", 0],
        filename_prefix: "ComfyUI_Music_App/song",
        format: { format: "mp3", quality: "320k" },
      },
    },
  };
}

async function queueMusic(settings, status, audioPlayer, downloadLink, button) {
  button.disabled = true;
  audioPlayer.hidden = true;
  downloadLink.hidden = true;
  status.textContent = "Enviando a música para a fila do ComfyUI…";
  try {
    const response = await api.fetchApi("/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: createWorkflow(settings),
        client_id: api.clientId,
      }),
    });
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.error?.message ?? "O ComfyUI recusou o fluxo.");
    }
    const result = await response.json();

    const deadline = Date.now() + maximumWait;
    status.textContent = `Na fila. ID da tarefa: ${result.prompt_id}`;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const historyResponse = await api.fetchApi(`/history/${encodeURIComponent(result.prompt_id)}`);
      if (!historyResponse.ok) throw new Error("Não foi possível consultar o resultado da geração.");
      const history = await historyResponse.json();
      const completed = history[result.prompt_id];
      if (!completed) continue;

      if (completed.status?.completed === false) {
        const messages = completed.status.messages ?? [];
        const error = messages.find(([type]) => type === "execution_error")?.[1];
        throw new Error(error?.exception_message ?? "O ComfyUI não conseguiu concluir a geração.");
      }

      const savedAudio = completed.outputs?.[outputNodeId]?.audio?.[0];
      if (!savedAudio) continue;
      const audioUrl = api.apiURL(`/view?${new URLSearchParams({
        filename: savedAudio.filename,
        subfolder: savedAudio.subfolder ?? "",
        type: savedAudio.type ?? "output",
      })}`);
      audioPlayer.src = audioUrl;
      audioPlayer.hidden = false;
      downloadLink.href = audioUrl;
      downloadLink.download = savedAudio.filename;
      downloadLink.hidden = false;
      status.textContent = `Pronto: ${savedAudio.filename}`;
      return;
    }
    throw new Error("A geração ainda não terminou. Confira a fila do ComfyUI.");
  } catch (error) {
    status.textContent = `Erro: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

function addStyles() {
  if (document.getElementById("comfy-music-app-styles")) return;
  const style = document.createElement("style");
  style.id = "comfy-music-app-styles";
  style.textContent = `
    .comfy-music-app { box-sizing: border-box; max-width: 760px; margin: 0 auto; padding: 24px; color: var(--fg-color, #eee); font: inherit; }
    .comfy-music-app *, .comfy-music-app *::before, .comfy-music-app *::after { box-sizing: border-box; }
    .comfy-music-app h2 { margin: 0 0 8px; font-size: 1.5rem; }
    .comfy-music-app p { line-height: 1.5; }
    .comfy-music-app .comfy-music-note { opacity: .78; font-size: .92rem; }
    .comfy-music-app .comfy-music-field { display: grid; gap: 7px; margin: 14px 0; }
    .comfy-music-app input, .comfy-music-app select, .comfy-music-app textarea { width: 100%; padding: 9px; color: inherit; background: var(--comfy-input-bg, #222); border: 1px solid var(--border-color, #555); border-radius: 6px; font: inherit; }
    .comfy-music-app textarea { min-height: 180px; resize: vertical; }
    .comfy-music-app .comfy-music-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0 14px; }
    .comfy-music-app .comfy-music-mode { display: flex; flex-wrap: wrap; gap: 16px; margin: 18px 0; }
    .comfy-music-app .comfy-music-mode label { display: flex; align-items: center; gap: 8px; }
    .comfy-music-app .comfy-music-mode input { width: auto; }
    .comfy-music-app button, .comfy-music-app a.comfy-music-download { display: inline-block; margin: 10px 10px 10px 0; padding: 10px 16px; border: 0; border-radius: 6px; color: var(--fg-color, #fff); background: var(--comfy-menu-bg, #3b78c8); font: inherit; font-weight: 600; cursor: pointer; text-decoration: none; }
    .comfy-music-app button:disabled { opacity: .55; cursor: wait; }
    .comfy-music-app [role="status"] { min-height: 1.5em; margin: 12px 0; }
    .comfy-music-app audio { display: block; width: 100%; margin: 18px 0; }
  `;
  document.head.append(style);
}

async function renderMusicApp(element) {
  addStyles();
  element.replaceChildren();
  const root = document.createElement("main");
  root.className = "comfy-music-app";
  root.innerHTML = `
    <h2>Estúdio de música</h2>
    <p class="comfy-music-note">Gere uma faixa com voz cantada usando o ACE-Step 1.5 já integrado ao ComfyUI.</p>
    <div class="comfy-music-mode" role="radiogroup" aria-label="Como preparar a letra">
      <label><input type="radio" name="comfy-music-mode" value="write" checked> Escrever minha letra</label>
      <label><input type="radio" name="comfy-music-mode" value="theme"> Criar uma letra simples a partir de um tema</label>
    </div>
    <section data-theme hidden></section>
    <section data-lyrics></section>
    <div class="comfy-music-grid">
      <div data-model></div><div data-clip1></div><div data-clip2></div><div data-vae></div>
    </div>
    <div class="comfy-music-grid">
      <div data-tags></div><div data-language></div><div data-timesignature></div><div data-keyscale></div>
      <div data-duration></div><div data-bpm></div><div data-seed></div>
    </div>
    <p class="comfy-music-note" data-info role="status" aria-live="polite">Verificando modelos ACE-Step locais…</p>
    <button type="button" data-generate disabled>Gerar música</button>
    <p data-status role="status" aria-live="polite"></p>
    <audio controls data-player hidden></audio>
    <a class="comfy-music-download" data-download hidden>Baixar música</a>
  `;
  element.append(root);

  const themeSection = root.querySelector("[data-theme]");
  const lyricsSection = root.querySelector("[data-lyrics]");
  const theme = field(themeSection, "Tema da música", textInput(""));
  theme.required = true;
  theme.placeholder = "Ex.: saudade de casa numa noite de verão";
  const draftButton = document.createElement("button");
  draftButton.type = "button";
  draftButton.textContent = "Montar rascunho da letra";
  themeSection.append(draftButton);
  const lyricTextarea = document.createElement("textarea");
  lyricTextarea.placeholder = "Digite ou revise a letra que será cantada…";
  lyricTextarea.maxLength = 5000;
  field(lyricsSection, "Letra (português ou outro idioma)", lyricTextarea);
  let draftedTheme;

  for (const radio of root.querySelectorAll('input[name="comfy-music-mode"]')) {
    radio.addEventListener("change", () => {
      const themeMode = radio.checked && radio.value === "theme";
      if (radio.checked) {
        themeSection.hidden = !themeMode;
        theme.required = themeMode;
      }
    });
  }

  const status = root.querySelector("[data-status]");
  const player = root.querySelector("[data-player]");
  const download = root.querySelector("[data-download]");
  const generateButton = root.querySelector("[data-generate]");
  draftButton.addEventListener("click", () => {
    if (!theme.reportValidity()) return;
    lyricTextarea.value = generatedLyrics(theme.value);
    draftedTheme = theme.value;
    status.textContent = "Rascunho local pronto. Revise a letra e clique em Gerar música.";
  });
  let models;

  try {
    const response = await api.fetchApi("/object_info");
    if (!response.ok) throw new Error("Não foi possível consultar os nós e modelos do ComfyUI.");
    models = await response.json();
  } catch (error) {
    root.querySelector("[data-info]").textContent = error.message;
    return;
  }

  const requiredNodes = [
    "UNETLoader", "DualCLIPLoader", "VAELoader", "TextEncodeAceStepAudio1.5",
    "ConditioningZeroOut", "EmptyAceStep1.5LatentAudio", "ModelSamplingAuraFlow",
    "KSampler", "VAEDecodeAudio", "SaveAudioAdvanced",
  ];
  const missingNodes = requiredNodes.filter((name) => !models[name]);
  if (missingNodes.length) {
    root.querySelector("[data-info]").textContent =
      `Este ComfyUI não oferece os nós ACE-Step necessários: ${missingNodes.join(", ")}. Atualize o ComfyUI.`;
    return;
  }

  const modelOptions = optionList(models.UNETLoader, "unet_name").filter((name) => /ace.?step|acestep/i.test(name));
  const clipOptions = optionList(models.DualCLIPLoader, "clip_name1");
  const vaeOptions = optionList(models.VAELoader, "vae_name").filter((name) => /ace.?1\.5.?vae|ace.?step/i.test(name));
  const clip2Options = optionList(models.DualCLIPLoader, "clip_name2");
  const clip1 = clipOptions.find((name) => /qwen_0\.6b_ace15/i.test(name)) ?? clipOptions.find((name) => /ace15|ace_step/i.test(name));
  const clip2 = clip2Options.find((name) => /qwen_4b_ace15/i.test(name)) ?? clip2Options.find((name) => /ace15|ace_step/i.test(name));
  const vae = vaeOptions.find((name) => /ace_1\.5_vae/i.test(name)) ?? vaeOptions[0];
  const model = modelOptions.find((name) => /acestep_v1\.5_turbo/i.test(name)) ?? modelOptions[0];

  if (!model || !clip1 || !clip2 || !vae) {
    root.querySelector("[data-info]").textContent =
      "Modelos ACE-Step ausentes. Instale localmente o diffusion model, os dois text encoders Qwen ACE 1.5 e o VAE ACE 1.5 nas pastas de modelos do ComfyUI; nenhum modelo será baixado pelo app.";
    return;
  }

  const modelSelect = field(root.querySelector("[data-model]"), "Modelo ACE-Step", selectInput(modelOptions, model));
  const clip1Select = field(root.querySelector("[data-clip1]"), "Text encoder menor", selectInput(clipOptions, clip1));
  const clip2Select = field(root.querySelector("[data-clip2]"), "Text encoder maior", selectInput(clip2Options, clip2));
  const vaeSelect = field(root.querySelector("[data-vae]"), "VAE de áudio", selectInput(vaeOptions, vae));
  const tags = field(root.querySelector("[data-tags]"), "Estilo e instrumentos", textInput("Brazilian pop, melodic, warm, sung vocals"));
  const language = field(root.querySelector("[data-language]"), "Idioma cantado", selectInput(optionList(models["TextEncodeAceStepAudio1.5"], "language"), "pt"));
  const timesignature = field(root.querySelector("[data-timesignature]"), "Compasso", selectInput(optionList(models["TextEncodeAceStepAudio1.5"], "timesignature"), "4"));
  const keyscale = field(root.querySelector("[data-keyscale]"), "Tonalidade", selectInput(optionList(models["TextEncodeAceStepAudio1.5"], "keyscale"), "C major"));
  const duration = field(root.querySelector("[data-duration]"), "Duração (segundos)", textInput("30", "number"));
  duration.required = true;
  duration.min = "1";
  duration.max = "600";
  duration.step = "1";
  const bpm = field(root.querySelector("[data-bpm]"), "BPM", textInput("120", "number"));
  bpm.required = true;
  bpm.min = "10";
  bpm.max = "300";
  const seed = field(root.querySelector("[data-seed]"), "Seed", textInput(String(Math.floor(Math.random() * 2147483647)), "number"));
  seed.required = true;
  seed.min = "0";
  seed.max = "4294967295";
  seed.step = "1";
  const info = root.querySelector("[data-info]");
  info.textContent = "Modelos ACE-Step encontrados localmente. A letra automática usa um modelo textual fixo local, não uma IA.";
  generateButton.disabled = false;

  generateButton.addEventListener("click", () => {
    const themeMode = root.querySelector('input[name="comfy-music-mode"]:checked').value === "theme";
    if (themeMode) {
      if (!theme.reportValidity()) return;
      if (draftedTheme !== theme.value) {
        lyricTextarea.value = generatedLyrics(theme.value);
        draftedTheme = theme.value;
        status.textContent = "Rascunho local pronto. Revise a letra e clique em Gerar música.";
        return;
      }
    } else if (!lyricTextarea.value.trim()) {
      lyricTextarea.setCustomValidity("Digite uma letra antes de gerar a música.");
      lyricTextarea.reportValidity();
      lyricTextarea.setCustomValidity("");
      return;
    }
    if (!duration.reportValidity() || !bpm.reportValidity() || !seed.reportValidity()) return;

    queueMusic({
      model: modelSelect.value,
      clip1: clip1Select.value,
      clip2: clip2Select.value,
      vae: vaeSelect.value,
      lyrics: lyricTextarea.value,
      tags: tags.value,
      language: language.value,
      timesignature: timesignature.value,
      keyscale: keyscale.value,
      duration: Number(duration.value),
      bpm: Number(bpm.value),
      seed: Number(seed.value),
    }, status, player, download, generateButton);
  });
}

app.registerExtension({
  name: extensionName,
  async setup() {
    app.extensionManager.registerSidebarTab({
      id: "comfy-music-app",
      icon: "pi pi-headphones",
      title: "Música",
      tooltip: "Estúdio de música",
      type: "custom",
      render: renderMusicApp,
    });
  },
});
