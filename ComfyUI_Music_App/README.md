# VØLTR 7 — Estúdio de música para ComfyUI

Extensão local e isolada que adiciona a aba **VØLTR 7** à barra lateral do ComfyUI. Usa o fluxo oficial ACE-Step 1.5 e os nós e a fila do próprio ComfyUI para gerar uma faixa com voz cantada; o resultado é reproduzido na aba e salvo em `output/ComfyUI_Music_App/` usando um nó de saída de áudio reconhecido pelo ComfyUI. O formato é MP3 quando disponível; se a instalação só tiver o nó legado FLAC, será salvo em FLAC.

## Instalação no ComfyUI Windows Portable

1. Copie a pasta `ComfyUI_Music_App` deste projeto para `D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\`.
2. Reinicie o ComfyUI e atualize o navegador. Abra a aba **VØLTR 7** na barra lateral. No terminal, a extensão deve carregar sem o aviso de `NODE_CLASS_MAPPINGS or comfy_entrypoint`.
3. Se algum modelo não aparecer, instale os arquivos ACE-Step 1.5 nas pastas de modelos do ComfyUI: diffusion model em `models\diffusion_models`, os text encoders Qwen em `models\text_encoders` e o VAE em `models\vae`. O app não baixa modelos nem instala dependências.

Para atualizar uma instalação existente, substitua a pasta `custom_nodes\ComfyUI_Music_App` pelos arquivos desta versão, reinicie o ComfyUI e recarregue a página. Idioma, compasso e tonalidade são preenchidos automaticamente com valores ACE-Step válidos (`pt`, `4`, `C major`). Se aparecer “Prompt sem saídas”, atualize também o ComfyUI e confira no terminal se os nós `SaveAudioAdvanced`, `SaveAudioMP3` ou `SaveAudio` estão disponíveis.

## Letras e voz

É possível escrever ou colar a letra, ou informar um tema. A opção por tema preenche a letra com uma estrutura curta fixa em português, gerada inteiramente no navegador — não é um modelo de linguagem nem uma letra criada por IA; revise o texto antes de gerar. A voz cantada e o acompanhamento dependem dos modelos locais ACE-Step 1.5 completos e dos nós nativos do ComfyUI. O ACE-Step também aceita letra vazia em outros fluxos do ComfyUI.

O app não faz chamadas a serviços externos e não modifica arquivos do núcleo do ComfyUI. Seu `comfy_entrypoint` registra uma extensão sem nós adicionais; a geração usa somente os nós nativos do ComfyUI.
