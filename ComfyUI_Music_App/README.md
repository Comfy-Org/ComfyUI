# Estúdio de música para ComfyUI

Extensão local e isolada que adiciona a aba **Música** à barra lateral do ComfyUI. Usa o fluxo oficial ACE-Step 1.5 e os nós e a fila do próprio ComfyUI para gerar uma faixa com voz cantada; o resultado é reproduzido na aba e salvo como MP3 em `output/ComfyUI_Music_App/`.

## Instalação no ComfyUI Windows Portable

1. Copie a pasta `ComfyUI_Music_App` deste projeto para `D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\`.
2. Reinicie o ComfyUI e atualize o navegador. Abra a aba **Música** na barra lateral.
3. Se algum modelo não aparecer, instale os arquivos ACE-Step 1.5 nas pastas de modelos do ComfyUI: diffusion model em `models\diffusion_models`, os text encoders Qwen em `models\text_encoders` e o VAE em `models\vae`. O app não baixa modelos nem instala dependências.

## Letras e voz

É possível escrever ou colar a letra, ou informar um tema. A opção por tema preenche a letra com uma estrutura curta fixa em português, gerada inteiramente no navegador — não é um modelo de linguagem nem uma letra criada por IA; revise o texto antes de gerar. A voz cantada e o acompanhamento dependem dos modelos locais ACE-Step 1.5 completos e dos nós nativos do ComfyUI. O ACE-Step também aceita letra vazia em outros fluxos do ComfyUI.

O app não faz chamadas a serviços externos e não modifica arquivos do núcleo do ComfyUI.
