# ComfyUI no Easypanel com NVIDIA

## Revisão e escopo

Base revisada: `b33e2b55cae074eca5aec96283cceac19aa249ba`, branch `master`.
Faltavam Dockerfile, exclusões do contexto, inicialização de container,
persistência, healthcheck e instruções de infraestrutura. Nenhum arquivo original
foi modificado. Não são necessários Redis, PostgreSQL ou outro serviço para o
ComfyUI básico: o banco SQLite fica no volume.

A imagem usa Linux x86_64, Python 3.12, PyTorch 2.12.1, torchvision 0.27.1 e CUDA
13.0. As versões do PyTorch ficam protegidas por constraints durante a instalação
do requirements original. As demais dependências seguem o upstream; não é um lock
completo. Guarde a imagem por digest para reproduzir e reverter um deploy.

**Pendente antes de produção:** confirmar modelo/VRAM da GPU, driver, distribuição,
RAM/disco e versão do Easypanel. Build Docker e inferência real precisam ser
validados na VPS; o ambiente usado para preparar esta alteração não tem Docker/GPU.

## 1. Preparar a VPS

A VPS precisa oferecer uma GPU NVIDIA de verdade, acessível ao sistema convidado.
Uma VPS apenas com CPU não ganha GPU pela instalação do Dockerfile.

No host Linux:

```sh
uname -m
nvidia-smi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
df -h
free -h
```

O perfil padrão atende GPUs compatíveis com CUDA 13 (por exemplo Turing e mais
novas). Use driver compatível da família 580 ou superior; confira a versão exata
suportada pelo hardware/provedor. GPUs antigas, como Pascal, precisam de outro
perfil: não use este padrão sem revisar a compatibilidade. Os argumentos de build
`TORCH_VERSION`, `TORCHVISION_VERSION` e `TORCH_CUDA` permitem selecionar um par
oficial compatível e reconstruir a imagem. Variáveis de runtime não trocam CUDA.

Instale o driver pelo procedimento da distribuição/provedor e o NVIDIA Container
Toolkit pelo [guia NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Não instale o driver dentro da imagem. Para o App Service/Swarm do Easypanel,
siga o [guia de GPU do Easypanel](https://easypanel.io/docs/guides/gpu-support):
runtime NVIDIA padrão, anúncio dos UUIDs como recursos genéricos e configuração
`swarm-resource` compatível com a versão instalada. Preserve as outras opções do
`daemon.json`; reiniciar Docker pode interromper os serviços da VPS.

O registro do runtime padrão pode ser feito após instalar o Toolkit:

```sh
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

Esse teste valida o acesso básico; não substitui o teste PyTorch dentro do serviço.
`NVIDIA_VISIBLE_DEVICES` seleciona dispositivos, mas não instala runtime/driver,
nem reserva a GPU exclusivamente. Em Swarm com vários nós, restrinja o serviço ao
nó com a GPU e o volume; configure as reservas conforme sua versão do painel.
Não transponha `--gpus all` de `docker run` para um campo de comando do aplicativo.

Planeje disco para imagem, cache do build, modelos e resultados. Modelos de vídeo
podem ocupar dezenas de GB cada. RAM e VRAM necessárias dependem do workflow;
um healthcheck saudável não comprova que determinado modelo cabe na GPU.

## 2. App Service no Easypanel

| Campo | Valor |
| --- | --- |
| Tipo | App |
| Source | GitHub: `lhresultadosdigitais-dev/ComfyUI` |
| Branch para testar esta alteração | `codex/easypanel-nvidia` |
| Branch depois de integrar o PR | `master` |
| Build Path | `/` |
| Builder / Dockerfile | Dockerfile / `Dockerfile` |
| Réplicas | `1` |
| Zero-downtime | Desativado: evita duas instâncias disputando VRAM e SQLite |
| Command override | Vazio; usa o CMD da imagem |
| Domain target | HTTP, porta `8188`, caminho `/` |
| Porta publicada no host | Não necessária; use o domínio/proxy |

Se precisar preencher o comando explicitamente:

```text
python /app/deploy/easypanel/start.py
```

O launcher termina com `exec`, substituindo-se pelo ComfyUI e permitindo receber
os sinais de parada. Ative Tini no painel se extensões criarem processos filhos.
Não configure atualização automática até validar o primeiro deploy.

### Volume obrigatório

Em Storage, crie um volume nomeado `comfyui-data` com Mount Path **`/data`**.
O Dockerfile não cria armazenamento persistente sozinho. Não monte `/app`.

| Diretório no volume | Conteúdo |
| --- | --- |
| `/data/models` | checkpoints, diffusion_models, text_encoders, vae, loras etc. |
| `/data/input` | imagens/áudios enviados |
| `/data/output` | resultados |
| `/data/user` | configurações, workflows salvos e `comfyui.db` |
| `/data/custom_nodes` | extensões adicionadas pelo operador |
| `/data/cache` | caches Hugging Face/PyTorch |

Temporários ficam em `/tmp/comfyui` e podem ser descartados. A fila em memória não
é recuperada após reinício. O volume precisa permitir escrita pelo usuário do
container (esta imagem usa root). Não monte Docker socket nem habilite privileged.
Os custom nodes originais em `/app/custom_nodes` continuam disponíveis; o YAML
adiciona `/data/custom_nodes` sem ocultá-los.

### Variáveis

Copie `.env.example` desta pasta para Environment. O arquivo não é carregado
sozinho. Não coloque tokens no repositório nem em argumentos do build.

- `COMFYUI_PORT=8188`: deve coincidir com a porta do domínio; o healthcheck usa este valor.
- `COMFYUI_ARGS=`: argumentos opcionais, por exemplo `--lowvram`. Aspas são aceitas;
  não há execução de shell. Endereço, porta e caminhos são gerenciados pelo launcher.
- `NVIDIA_VISIBLE_DEVICES=all`: em servidor compartilhado, prefira o UUID selecionado.
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`: CUDA e diagnóstico NVIDIA.
- `HF_HOME` e `TORCH_HOME`: caches no volume, já definidos na imagem.

Os argumentos de build têm os padrões declarados no Dockerfile. Para alterá-los,
use o mecanismo de build args suportado pela sua versão do painel, ou construa
fora dele e publique a imagem em um registro. Não basta editar Environment e
presumir que qualquer versão do painel encaminhe build args automaticamente.

### Acesso

Configure DNS, HTTPS e Basic Auth na aba Security antes de expor o domínio.
Proteja também o domínio automático do serviço. Verifique o acesso autenticado
à interface, API e WebSocket `/ws`. `--multi-user` não substitui autenticação.
Não habilite CORS global para resolver erros do proxy. Clientes de API precisam
usar as credenciais do proxy; mantenha-as fora dos workflows compartilhados.

## 3. Validar

O build instala requisitos e executa `pip check`; não exige GPU no builder.
O healthcheck consulta `/system_stats` localmente a cada 30 s, com timeout de 5 s,
janela inicial de 180 s e 5 tentativas. Não executa inferência e não depende do DNS
ou da autenticação do domínio. Inicializações muito lentas podem exigir ampliar
`--start-period` no Dockerfile. Examine os logs antes de mudar o limite.

Na Shell do serviço:

```sh
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.ones((32,32),device='cuda'); print((x@x).sum().item()); torch.cuda.synchronize()"
python -c "import os,urllib.request; print(urllib.request.urlopen('http://127.0.0.1:'+os.environ['COMFYUI_PORT']+'/system_stats').status)"
python -m pip check
```

Depois envie um arquivo, salve um workflow, execute uma geração com os modelos
necessários e faça um redeploy. Confirme que upload, workflow, modelo e resultado
continuam presentes e que a GPU foi usada. Teste também o domínio sem credenciais:
a interface e a API não devem ficar acessíveis.

Para testar fora do painel, no host GPU e a partir da raiz do repositório:

```sh
docker build -t comfyui:easypanel .
docker run -d --name comfyui-test --gpus all --mount source=comfyui-test-data,target=/data -p 127.0.0.1:8188:8188 comfyui:easypanel
docker logs comfyui-test
docker inspect --format '{{.State.Health.Status}}' comfyui-test
```

## 4. Modelos, extensões e manutenção

A imagem não inclui pesos nem baixa modelos ao iniciar. Coloque os arquivos
exigidos pelo workflow nas subpastas corretas de `/data/models`, criando-as
quando necessário. Respeite as licenças dos modelos escolhidos.

Custom nodes são código executável. Instale somente os necessários em
`/data/custom_nodes`, evitando duplicar nomes dos nodes originais. Bibliotecas
instaladas manualmente com pip na Shell somem no redeploy: acrescente dependências
em uma imagem derivada, com versões controladas e as constraints do PyTorch.
Algumas extensões exigem compiladores/CUDA de desenvolvimento que esta imagem não
inclui. ComfyUI Manager não é instalado ou habilitado automaticamente.

Faça backup do volume, sobretudo `user`, `input`, `output` e extensões; inclua pesos
se não puder baixá-los novamente. Pare o serviço para obter cópia consistente do
SQLite, ou use uma ferramenta de backup SQLite. Teste a restauração. Antes de
atualizar, guarde imagem/digest e backup: reverter só a imagem pode não reverter
migrações do banco. Use um volume separado ao testar versões novas.

| Sintoma | Verificação |
| --- | --- |
| GPU não encontrada | GPU no host, driver, Toolkit, runtime do Swarm, UUID e nó correto |
| Driver insuficiente / no kernel image | compatibilidade de driver, GPU e wheel CUDA |
| 502 | logs, bind 0.0.0.0, porta do domínio e healthcheck |
| CUDA out of memory | modelo/resolução/batch, outros processos na GPU; avaliar `--lowvram` |
| Model not found | nomes e subpastas dos pesos do workflow |
| Dados perdidos | volume `/data`, mesmo nó e mesmo volume após deploy |
| Import error em custom node | instalar dependências na imagem e reconstruir |

Referências: [Easypanel App](https://easypanel.io/docs/services/app),
[pares oficiais PyTorch](https://pytorch.org/get-started/previous-versions/),
[compatibilidade CUDA/driver](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
e o README original deste repositório.
