const starterWords = [
  { word: "serendipity", meaning: "意外发现美好事物的能力", example: "It was pure serendipity that we met at the little bookstore.", phonetic: "/ˌser.ənˈdɪp.ə.ti/" },
  { word: "bloom", meaning: "开花；焕发活力", example: "Give yourself time to bloom.", phonetic: "/bluːm/" },
  { word: "wander", meaning: "漫步；走走停停", example: "We wandered through the quiet streets.", phonetic: "/ˈwɑːn.dɚ/" },
  { word: "resilient", meaning: "有韧性的；能恢复的", example: "Small habits help us become more resilient.", phonetic: "/rɪˈzɪl.jənt/" },
  { word: "notion", meaning: "想法；概念", example: "I like the notion of starting again.", phonetic: "/ˈnoʊ.ʃən/" },
  { word: "delight", meaning: "使愉快；欣喜", example: "The smallest things can bring delight.", phonetic: "/dɪˈlaɪt/" },
  { word: "gentle", meaning: "温柔的；轻柔的", example: "Be gentle with yourself today.", phonetic: "/ˈdʒen.t̬əl/" },
  { word: "curiosity", meaning: "好奇心", example: "Curiosity is a lovely place to begin.", phonetic: "/ˌkjʊr.iˈɑː.sə.t̬i/" },
];

const courseWeeks = [
  {
    phase: "第一阶段 · 敢开口", title: "先听见英语的声音", goal: "认识字母与常见发音，学会打招呼和介绍自己。",
    lessons: [
      ["字母和名字", "读一遍 A–Z 字母名，重点分清容易听混的 B / P、G / J。", "Hi, I'm ___.（你好，我是……）"],
      ["五个元音", "认识 a、e、i、o、u 的常见短音；慢慢读 cat、pen、sit、hot、bus。", "cat · pen · sit · hot · bus"],
      ["拼写自己的名字", "练习说名字里的字母，再把自己的名字拼给自己听。", "My name is ___. It's spelled ___.（我叫……，拼作……）"],
      ["打招呼和告别", "选三种问候语，分别练习早上见面、第一次见面和告别。", "Hello. / Nice to meet you. / See you."],
      ["第一段自我介绍", "用名字、来自哪里、问候语，录下或写下 3 句自我介绍。", "Hi, I'm ___. I'm from ___. Nice to meet you."],
    ],
  },
  {
    phase: "第一阶段 · 敢开口", title: "说说你是谁", goal: "掌握人称代词和 be 动词，能问候、回答简单问题。",
    lessons: [
      ["我、你、他、她", "读熟 I、you、he、she、we、they，并各配一个认识的人。", "I = 我 · you = 你 · she = 她"],
      ["I'm / you're", "练习 I'm 和 you're 的缩写，用名字和身份各造两句。", "I'm Y. You're my friend.（我是 Y。你是我的朋友。）"],
      ["他和她", "用 he 描述一位男性、用 she 描述一位女性；注意 be 动词用 is。", "He is kind. She is a student."],
      ["问候与回答", "大声问答三轮：名字、近况和来自哪里。", "How are you? — I'm good, thanks."],
      ["复习：认识新朋友", "扮演初次见面，用 4 句完成问候和自我介绍。", "Hello! What's your name? I'm ___. Nice to meet you."],
    ],
  },
  {
    phase: "第一阶段 · 敢开口", title: "把简单句说完整", goal: "理解 am / is / are，学会肯定句、否定句和 yes/no 问句。",
    lessons: [
      ["am、is、are", "把 I、he/she、you/we/they 分别和 am、is、are 配对读三遍。", "I am ready. She is here. They are happy."],
      ["说“不”", "在 be 动词后加 not，把三句肯定句改成否定句。", "I'm not tired. He isn't at home."],
      ["问一个是非题", "把 is/are 放到句首，练习提问并用 yes 或 no 回答。", "Are you ready? — Yes, I am."],
      ["问名字和身份", "练习 What's ...? 和 Who is ...?，回答时用完整短句。", "Who is she? — She is my friend."],
      ["复习：介绍身边的人", "介绍自己和一位朋友：姓名、身份、感受各一句。", "This is ___. She is my friend. We are happy."],
    ],
  },
  {
    phase: "第二阶段 · 组成句子", title: "身边的人和东西", goal: "会用 a / an、名词复数和 this / that 描述常见事物。",
    lessons: [
      ["a 和 an", "给身边 5 个东西加 a 或 an；留意 an 用在元音音素开头前。", "a book · an apple · a cup"],
      ["一个还是多个", "把 book、pen、apple 变成复数，给房间里的物品数数。", "one book, two books · one box, two boxes"],
      ["this 和 that", "指近处和远处的东西，分别用 this 和 that 造句。", "This is a cup. That is a window."],
      ["these 和 those", "拿起几件物品练习 these；指远处的多个物品用 those。", "These are my keys. Those are books."],
      ["复习：我的桌面", "用 5 句介绍桌面上的物品，至少用一次复数句。", "This is a pen. These are my books."],
    ],
  },
  {
    phase: "第二阶段 · 组成句子", title: "每天会做什么", goal: "学会常见动词和一般现在时，描述日常习惯。",
    lessons: [
      ["日常动作动词", "学习 get up、eat、go、work、study、sleep，用动作记词。", "I study English.（我学英语。）"],
      ["I / you 的日常句", "选 3 个日常动作，用 I 和 you 分别说完整句子。", "I eat breakfast. You go to work."],
      ["he / she 加 s", "把 I 的习惯改成家人或朋友的习惯；留意动词词尾变化。", "I like tea. She likes tea."],
      ["说频率", "用 every day、often、sometimes 描述一周的习惯。", "I read every day. I sometimes cook."],
      ["复习：我的一天", "按起床、白天、晚上顺序，说 4 句自己的日常。", "I get up at ___. I study in the evening."],
    ],
  },
  {
    phase: "第二阶段 · 组成句子", title: "提问、喜好和时间", goal: "用 do / does 提问，表达喜欢什么以及简单作息。",
    lessons: [
      ["喜欢与不喜欢", "用 like / don't like 列出三样喜欢和一样不喜欢的东西。", "I like coffee. I don't like cold weather."],
      ["Do you...? 问问题", "用 Do you...? 问三件事，练习 Yes, I do / No, I don't。", "Do you like music? — Yes, I do."],
      ["Does he/she...?", "为朋友或家人提问；does 后面的动词用原形。", "Does she work here? — No, she doesn't."],
      ["时间和星期", "说出今天星期几，并用 at 说一个日常时间。", "It's Monday. I get up at seven."],
      ["复习：互相了解", "写下或说出 3 个问题和回答，至少有一个 does 问句。", "What do you like? Do you study English every day?"],
    ],
  },
  {
    phase: "第二阶段 · 组成句子", title: "家人、朋友和拥有的东西", goal: "掌握 my / your 等物主词和常见宾语代词。",
    lessons: [
      ["我的和你的", "用 my、your、his、her 介绍姓名、家人或物品。", "This is my sister. Her name is ___. "],
      ["家人词汇", "选 5 个家庭成员词，给每个人配一句简单介绍。", "I have a brother. He is kind."],
      ["have 和 has", "说说自己和一位朋友拥有什么；he/she 用 has。", "I have a bike. She has a cat."],
      ["me、him、her", "学习 I see him / She knows me 这类句子，分清主语和宾语。", "I like her. She likes me."],
      ["复习：介绍我的朋友", "介绍一个朋友：姓名、关系、一件拥有的东西和一个喜好。", "This is my friend. He has a dog. He likes music."],
    ],
  },
  {
    phase: "第三阶段 · 日常交流", title: "描述位置和身边环境", goal: "用 there is / are 和常见介词描述房间、街道与地点。",
    lessons: [
      ["房间里的位置", "用 in、on、under 描述三个物品现在的位置。", "The book is on the table."],
      ["there is：一个", "看房间或想象一个房间，说出里面有什么。", "There is a chair by the window."],
      ["there are：多个", "用 there are 数一数房间里的物品，练习单复数一致。", "There are two cups on the table."],
      ["问路的地点词", "学习 near、next to、across from，描述家附近的地方。", "The café is next to the bank."],
      ["复习：带人逛房间", "用 5 句话介绍房间和物品位置，至少使用一个 there are。", "There is a desk. The lamp is on the desk."],
    ],
  },
  {
    phase: "第三阶段 · 日常交流", title: "会请求，也能点餐", goal: "使用 can / can't 表达能力、请求和简单的礼貌需求。",
    lessons: [
      ["can 表示会做", "用 can 说出自己会做的两件事和暂时不会的一件事。", "I can cook. I can't drive."],
      ["Can you...? 提问", "问别人会不会做某事，并用 Yes, I can / No, I can't 回答。", "Can you help me? — Yes, I can."],
      ["礼貌地请求", "把 Can I have...? 用于饮品、食物或需要的物品。", "Can I have some water, please?"],
      ["咖啡店点单", "练习问候、点饮料、说谢谢，读一遍完整小对话。", "I'd like a tea, please. — Here you are."],
      ["复习：点一份早餐", "自己扮演顾客和店员，完成 4 轮点单对话。", "Can I have an egg and some tea, please?"],
    ],
  },
  {
    phase: "第三阶段 · 日常交流", title: "说说正在发生的事", goal: "初步认识现在进行时，描述眼前的活动和天气。",
    lessons: [
      ["动作 + ing", "学习 read→reading、make→making 等形式，边做动作边说。", "I am reading.（我正在读书。）"],
      ["I'm doing...", "描述自己和身边的人此刻正在做什么。", "She is cooking. They are talking."],
      ["现在进行时提问", "练习 What are you doing? 并用一句话回答。", "What are you doing? — I'm learning English."],
      ["天气和衣物", "学习 sunny、rainy、cold、warm，并说适合穿什么。", "It's rainy. I'm wearing a coat."],
      ["复习：现场播报", "像播报员一样，说说天气和两个人正在做的事。", "It's sunny. I am walking. My friend is taking a photo."],
    ],
  },
  {
    phase: "第四阶段 · 建立信心", title: "讲讲刚刚发生的事", goal: "认识 was / were 和常用过去式，能说简单的昨天。",
    lessons: [
      ["was 和 were", "把 am/is 换成 was，把 are 换成 were，描述昨天的状态。", "I was at home. They were happy."],
      ["昨天做了什么", "学习 went、had、ate、saw 四个常用不规则过去式。", "I went to work. I had lunch."],
      ["规则动词加 ed", "把 walk、watch、cook 变成过去式，说说昨天做了什么。", "I watched a movie last night."],
      ["询问昨天", "练习 What did you do...?；did 后面的动词用原形。", "What did you do yesterday? — I cooked dinner."],
      ["复习：我的昨天", "用 4 句话写或说昨天；至少包含一个 was/were 和两个动作。", "I was busy. I worked and cooked dinner."],
    ],
  },
  {
    phase: "第四阶段 · 建立信心", title: "把学过的用起来", goal: "综合运用基础句型，完成简短自我介绍和生活对话。",
    lessons: [
      ["30 秒自我介绍", "介绍姓名、所在城市、工作或学习，以及一个喜欢的事物。", "I'm ___. I live in ___. I like ___. "],
      ["一问一答小对话", "练习问候、问近况、询问喜好，并自然地结束对话。", "How are you? What do you like? See you soon!"],
      ["读懂一段短文", "写 4 句介绍自己的一天，圈出动词并逐句理解，不求速度。", "I get up at seven. I have breakfast at home."],
      ["写 5 句小日记", "用今天、喜欢、正在做的事写 5 句；可以看词库，不必查语法。", "Today is ___. I feel ___. I am learning English."],
      ["我的英语第一站", "不看提示完成自我介绍，再写下最熟悉的 5 个句型作为回顾。", "I can introduce myself in English.（我可以用英语介绍自己。）"],
    ],
  },
];

const resourceTracks = [
  {
    id: "nce1", title: "新概念英语 1", level: "入门 · 基础句型", duration: "建议 12–16 周",
    description: "配合第一册学习日常词汇、基本语序与常用时态，练习把短句说完整。",
    units: [
      ["起步：发音与短句", "字母音、常见拼读、问候与介绍", "听读对应课文；圈出熟悉的词；跟读并录下三句自我介绍。"],
      ["人物与物品", "人称代词、a/an、单复数、指示词", "每课挑 5 个名词做卡片，用 this/these 描述身边物品。"],
      ["be 动词与描述", "am/is/are、否定句、一般疑问句", "把课文中的 be 句改成肯定、否定和提问三种形式。"],
      ["日常动作", "一般现在时、三单、频率副词", "找出日常动作词，替换主语说说自己和家人的习惯。"],
      ["提问与回答", "what/who/where、do/does 问句", "从课文内容自拟 5 个简单问题，并用完整句回答。"],
      ["时间与地点", "时间表达、介词、there is/are", "用 4 句介绍一天的时间安排和一个熟悉的地点。"],
      ["过去与将来", "常用过去式、be going to", "做一张不规则动词小卡；各说两句昨天和明天的安排。"],
      ["第一册复盘", "整合核心句型、听读与口头表达", "选已学材料复听复述；完成一段 60 秒日常生活介绍。"],
    ],
  },
  {
    id: "nce2", title: "新概念英语 2", level: "初中级 · 叙事表达", duration: "建议 16–24 周",
    description: "用短篇叙事巩固时态、从句和连贯表达；教材章节请按自己的版本对应。",
    units: [
      ["建立课文学习法", "分段听读、抓主旨、整理高频搭配", "每篇先不查词听/读一遍；用一句中文概括，再列 5 个有用搭配。"],
      ["讲清过去的故事", "一般过去时、过去进行时、时间顺序", "把一个故事按开端、经过、结果写成 5 句。"],
      ["经历与变化", "现在完成时、for/since、already/yet", "整理现在完成时例句，并比较“过去某时”与“到现在”的意思。"],
      ["未来与计划", "will、be going to、时间状语从句", "写下三个近期计划，并口头说明原因和时间。"],
      ["让句子更有层次", "宾语从句、关系代词、连接词", "从课文找复合句，划分主句和从句，再仿写两句。"],
      ["描述、比较与强调", "形容词、副词、比较级和最高级", "用同一话题写三种比较句，检查 than 与 the 的位置。"],
      ["间接表达与被动", "间接引语、被动语态入门", "将两句直接引语改述；找出被动句中动作的承受者。"],
      ["听读与复述", "关键词笔记、语音语调、短篇复述", "听/读一篇熟悉课文，记录 5 个关键词并脱稿复述 1 分钟。"],
      ["写作迁移", "段落结构、代词指代、衔接", "围绕熟悉话题写 80–100 词短文，检查时态和连接词。"],
      ["综合复习", "错题回看、句型主动提取", "不看笔记讲述一篇故事；把仍会出错的 10 个表达收进复习表。"],
    ],
  },
  {
    id: "nce3", title: "新概念英语 3", level: "中高级 · 阅读与写作", duration: "建议 20–30 周",
    description: "以较长文章为材料，提升复杂句分析、语篇逻辑和准确改写能力。",
    units: [
      ["长文阅读策略", "段落主旨、论点与例证、上下文猜词", "先概括每段功能；查词前标出转折、因果和举例信号词。"],
      ["拆解复杂句", "主干定位、定语从句、名词性从句", "为长句标出主谓宾和从句边界，再用简单英语改写句意。"],
      ["非谓语与压缩表达", "不定式、动名词、分词短语", "比较同一意思的从句与分词表达，写出各自适合的语境。"],
      ["虚拟与假设", "条件句、wish、非真实语气", "按“现实/假设/结果”列句子框架，自己写 4 个假设句。"],
      ["语气与强调", "倒装、强调结构、情态与推测", "在文章中找语气较强的句子，改写成中性表达并比较语气。"],
      ["语篇衔接", "指代、替换、省略、连接手段", "给段落中的代词标出指代对象，整理表达因果与转折的方式。"],
      ["精读与词汇笔记", "词族、搭配、语域和释义辨析", "每篇只收集 8 个值得复用的词组，给每个词组写原创例句。"],
      ["仿写与编辑", "主题句、展开、句间逻辑、校对", "写 120–150 词段落，分别检查内容、结构、语法和用词。"],
      ["听读复述", "信息分层、语速适应、摘要", "先听/读获取主旨，再复述论点和两条支持细节。"],
      ["综合迁移", "阅读、表达与自我纠错", "选择一个熟悉话题，引用阅读中学到的结构完成口头摘要和短文。"],
    ],
  },
  {
    id: "cambridge-basic", title: "剑桥英语语法 · 初级", level: "基础语法补强", duration: "建议 8–12 周",
    description: "按语法主题补牢基础；使用手头的 Cambridge 初级语法材料完成对应练习。",
    units: [
      ["句子和 be 动词", "主语、be、肯定/否定/提问", "用自己的信息各写 5 个肯定句、否定句和一般疑问句。"],
      ["一般现在时", "动词变化、三单、频率副词", "写一周日常安排；圈出 he/she 主语并核对动词词尾。"],
      ["现在进行时", "be + -ing、此刻与习惯的区别", "描述图片或房间里正在发生的事，写 6 句。"],
      ["一般过去时", "规则与常见不规则动词、did 问句", "用时间线写昨天发生的 5 件事，再改写成问句。"],
      ["将来表达", "will、going to、常见安排", "分别用两种结构表达预测、决定和已安排的事情。"],
      ["名词与冠词", "可数/不可数、a/an/the、复数", "给日常购物清单分类，解释每项前为什么用该冠词。"],
      ["代词与形容词", "物主词、宾格、比较级基础", "介绍家人和物品；用 3 组比较句描述偏好。"],
      ["介词与数量", "时间/地点介词、some/any/much/many", "描述房间和一天的安排，检查介词和量词搭配。"],
      ["情态动词与祈使句", "can、must、should、请求与建议", "写一张简单的出行建议清单，语气礼貌清楚。"],
      ["基础语法总复习", "时态、问句、词类、常见错误", "挑自己最常错的 10 个句子，先改正，再解释原因。"],
    ],
  },
  {
    id: "cambridge-intermediate", title: "剑桥英语语法 · 中级", level: "中级语法与准确度", duration: "建议 12–18 周",
    description: "系统复习核心时态、情态、从句与语态，重点练习语境中的选择。",
    units: [
      ["现在时态对比", "一般现在、现在进行、状态动词", "用同一主题各写三种句子，说明习惯、正在发生和长期状态。"],
      ["完成时态", "现在完成、完成进行、for/since", "按时间线比较经历、结果和持续时间；造 6 个个人例句。"],
      ["过去时态组合", "过去进行、过去完成、叙事顺序", "重写一个小故事，体现背景、先发生的事和主要事件。"],
      ["未来形式", "will、going to、进行时表安排", "给每句话标注预测、意图、临时决定或确定安排。"],
      ["情态与推测", "义务、许可、建议、可能性", "用 must/might/can't 对同一情境表达不同把握程度。"],
      ["被动语态", "时态变化、施事省略、使用场景", "选一个流程写 5 个被动句，并判断何时主动更自然。"],
      ["间接引语", "陈述、问题、时态回移", "把一段简短对话改述为间接引语，核对人称与时间词。"],
      ["关系从句", "限定/非限定从句、关系代词", "合并短句并判断信息是否必要，练习逗号的作用。"],
      ["条件句与假设", "零/一/二/三条件句、混合语境", "按真实程度改写同一个情境，说明事实与假设的区别。"],
      ["动名词与不定式", "动词搭配、意义差异", "把常见动词分类，给每类写自己的搭配例句。"],
      ["冠词、限定词与量词", "the/零冠词、限定词、数量表达", "从自写段落中标出名词短语，检查冠词和数量词。"],
      ["介词与衔接", "常见介词搭配、连接副词、复盘", "编辑一篇短文，提升介词搭配并减少重复连接词。"],
    ],
  },
  {
    id: "cambridge-advanced", title: "剑桥英语语法 · 高级", level: "高级语法与语体", duration: "建议 16–24 周",
    description: "面向高阶学习者，在真实表达中掌握复杂结构、语气、焦点和语体选择。",
    units: [
      ["时态与体的精细区别", "完成体、进行体、时间视角", "对照上下文说明时态选择如何改变时间范围和叙述重点。"],
      ["情态、推断与立场", "情态完形、可能性等级、委婉语气", "对同一事实写出确定、谨慎和委婉三种表达。"],
      ["高级条件与假设", "混合条件、倒装条件、隐含条件", "将 if 句改写为不同语序，比较正式程度与含义。"],
      ["强调与信息焦点", "倒装、分裂句、前置与后置", "改写句子以突出时间、人物或原因，并确保自然清楚。"],
      ["分词与压缩从句", "分词从句、独立结构、逻辑主语", "压缩一段重复短句，再检查分词动作的执行者是否明确。"],
      ["高级名词从句与关系结构", "嵌套从句、名词补语、复杂关系结构", "标注长句层级，将过度复杂的句子拆分并保持逻辑。"],
      ["动词模式与意义差别", "动名词/不定式、使役、感官动词", "用例句比较结构选择带来的含义或视角差异。"],
      ["冠词与语境含义", "泛指/特指、可数性、抽象名词", "编辑一段说明文，解释关键名词短语中的冠词选择。"],
      ["介词与固定搭配", "动词/形容词搭配、复杂介词", "将搭配放进完整语境，而不是孤立背译文。"],
      ["语篇与信息结构", "指代链、替换、省略、主题推进", "分析段落如何引入新信息，再仿写连贯的原创段落。"],
      ["语域与语气调整", "正式/非正式、缓和、礼貌与准确", "把一封随意消息改写为得体的正式邮件，并说明改动。"],
      ["高级编辑与综合复盘", "歧义、平行结构、句子节奏、自我校对", "校对一篇 200 词原创短文，按优先级记录反复出现的错误。"],
    ],
  },
  {
    id: "ielts", title: "雅思真经 · 备考路线", level: "听说读写 · 考试训练", duration: "建议 12–16 周",
    description: "使用自己持有的雅思真经及官方材料练习；网站提供原创训练节奏，不包含真题或答案。",
    units: [
      ["水平诊断与目标", "了解考试结构、目标分数和可投入时间", "做一组限时诊断；记录各科得分、耗时和最明显的失分原因。"],
      ["听力：定位信息", "读题预测、同义替换、数字拼写", "使用手头练习做一段精听；记录漏听处及导致失分的原因。"],
      ["听力：跟上语篇", "场景词、转折信号、笔记与检查", "练习按题目顺序定位信息；复听错误段并复述关键信息。"],
      ["阅读：定位与同义改写", "略读、扫读、关键词与段落主旨", "先限时完成一篇练习，再标注答案依据和原文同义表达。"],
      ["阅读：题型策略", "判断、匹配、填空的指令与证据", "按题型分组练习；每题指出原文证据，避免只凭印象作答。"],
      ["写作 Task 1", "识别图表/流程/地图任务、概述与比较", "用自己的练习材料写一段概述；优先检查主要特征是否选对。"],
      ["写作 Task 2：观点结构", "审题、立场、主体段与论据", "用 5 分钟列提纲，再写两段主体；检查每段是否支持中心观点。"],
      ["写作：语言与复盘", "衔接、准确用词、复杂句与自我编辑", "重写一篇旧作文，优先修复任务回应、逻辑和反复语法错误。"],
      ["口语 Part 1 与 Part 2", "自然扩展、具体细节、限时独白", "录音回答熟悉话题；回听并找出停顿、重复和可改进的发音。"],
      ["口语 Part 3 与互动", "解释原因、比较观点、谨慎表达", "针对一个社会话题练习观点—理由—例子，并尝试回应反方。"],
      ["限时模考与错因分析", "时间分配、精力管理、错误分类", "用持有的合规练习材料完成一次限时训练，按题型整理错因。"],
      ["考前巩固与个人策略", "高频错误清单、节奏、复习取舍", "重做最薄弱题型，整理个人考场流程；不在最后阶段盲目刷量。"],
    ],
  },
];

const storageKey = "daylight-english-space";
const state = loadState();
let reviewQueue = state.words.slice(0, 5);
let reviewIndex = 0;
let isRevealed = false;
let reviewedToday = 0;
let openCourseWeek = 0;
let currentAccount = null;
let authMode = "login";
let registrationAvailable = false;
let backendAvailable = null;
let syncTimer = null;
const accountMarkerKey = `${storageKey}-account`;
const guestBackupKey = `${storageKey}-guest-backup`;
const retriedWords = new Set();

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    return {
      words: Array.isArray(saved.words) ? saved.words : starterWords.map((word) => ({ ...word })),
      tasks: saved.tasks && saved.tasks.date === dateKey() ? saved.tasks.items : {},
      writing: saved.writing || "",
      activeDates: Array.isArray(saved.activeDates) ? saved.activeDates : [],
      completedLessons: Array.isArray(saved.completedLessons) ? saved.completedLessons : [],
      selectedCourse: typeof saved.selectedCourse === "string" ? saved.selectedCourse : "foundation",
    };
  } catch (error) {
    backendAvailable = false;
    console.warn("Could not load the saved English practice data.", error);
    return { words: starterWords.map((word) => ({ ...word })), tasks: {}, writing: "", activeDates: [], completedLessons: [], selectedCourse: "foundation" };
  }
}

function serializedState() {
  return {
    words: state.words,
    tasks: { date: dateKey(), items: state.tasks },
    writing: state.writing,
    activeDates: state.activeDates,
    completedLessons: state.completedLessons,
    selectedCourse: state.selectedCourse,
  };
}

function writeLocalState() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(serializedState()));
  } catch (error) {
    console.warn("Could not save the English practice data.", error);
    setSyncStatus("本机保存失败", "sync-error");
  }
}

function saveState() {
  writeLocalState();
  if (currentAccount && !currentAccount.offline) {
    clearTimeout(syncTimer);
    syncTimer = setTimeout(() => { flushSync(); }, 650);
    setSyncStatus("正在保存…", "syncing");
  }
}

function dateKey(now = new Date()) {
  return `${now.getFullYear()}-${now.getMonth() + 1}-${now.getDate()}`;
}

function dateOffset(key, days) {
  const [year, month, day] = key.split("-").map(Number);
  return dateKey(new Date(year, month - 1, day + days, 12));
}

async function requestApi(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  const response = await fetch(path, {
    ...options,
    headers,
    credentials: "same-origin",
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    const error = new Error(payload.error || "The server could not complete this request.");
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function mergeSerializedStates(localData, remoteData) {
  const local = localData || {};
  const remote = remoteData || {};
  const words = new Map();
  for (const item of [...(Array.isArray(remote.words) ? remote.words : []), ...(Array.isArray(local.words) ? local.words : [])]) {
    if (item && typeof item.word === "string") words.set(item.word.trim().toLowerCase(), item);
  }
  const currentTasks = (value) => value?.items
    ? value.date === dateKey() ? value.items : {}
    : value || {};
  const localTasks = currentTasks(local.tasks);
  const remoteTasks = currentTasks(remote.tasks);
  const taskKeys = new Set([...Object.keys(remoteTasks), ...Object.keys(localTasks)]);
  const items = Object.fromEntries([...taskKeys].map((key) => [key, Boolean(remoteTasks[key] || localTasks[key])]));
  const localWriting = typeof local.writing === "string" ? local.writing : "";
  const remoteWriting = typeof remote.writing === "string" ? remote.writing : "";
  const writing = localWriting.trim().split(/\s+/).filter(Boolean).length >= remoteWriting.trim().split(/\s+/).filter(Boolean).length
    ? localWriting : remoteWriting;
  return {
    words: [...words.values()],
    tasks: { date: dateKey(), items },
    writing,
    activeDates: [...new Set([
      ...(Array.isArray(remote.activeDates) ? remote.activeDates : []),
      ...(Array.isArray(local.activeDates) ? local.activeDates : []),
    ])],
    completedLessons: [...new Set([
      ...(Array.isArray(remote.completedLessons) ? remote.completedLessons : []),
      ...(Array.isArray(local.completedLessons) ? local.completedLessons : []),
    ])],
    selectedCourse: local.selectedCourse || remote.selectedCourse || "foundation",
  };
}

function applySerializedState(saved) {
  const merged = mergeSerializedStates(saved, {});
  Object.assign(state, {
    words: merged.words,
    tasks: merged.tasks.items,
    writing: merged.writing,
    activeDates: merged.activeDates,
    completedLessons: merged.completedLessons,
    selectedCourse: merged.selectedCourse,
  });
  reviewQueue = state.words.slice(0, 5);
  reviewIndex = 0;
  isRevealed = false;
  retriedWords.clear();
  document.querySelector("#writing-input").value = state.writing;
  renderApplication();
}

function renderApplication() {
  const streak = currentStreak();
  document.querySelector("#streak-count").textContent = streak;
  document.querySelector("#streak-stat").textContent = streak;
  renderStreakWeek();
  renderWords(document.querySelector("#word-search").value);
  renderTasks();
  renderCourse();
  updateReviewStats();
  updateWritingCount();
}

function setSyncStatus(message, status = "") {
  const statusElement = document.querySelector("#sync-status");
  statusElement.textContent = message;
  statusElement.className = `sync-status ${status}`.trim();
  document.querySelector("#account-status").textContent = currentAccount
    ? status === "sync-error" ? "同步遇到问题 · 点击查看" : status === "syncing" ? "正在同步学习数据…" : "账号已同步 · 点击管理"
    : message === "服务器不可用" ? "启动个人服务器以登录同步" : "本机保存 · 点击登录同步";
}

function updateAccountControls() {
  const signedIn = Boolean(currentAccount && !currentAccount.offline);
  const setupRequired = !currentAccount && backendAvailable === false;
  const accountTabs = document.querySelector("#account-tabs");
  const accountForm = document.querySelector("#account-form");
  const accountUser = document.querySelector("#account-user");
  const registerTab = document.querySelector("#register-tab");
  const accountError = document.querySelector("#account-error");
  document.querySelector("#account-name").textContent = currentAccount ? currentAccount.username : "我的学习角";
  document.querySelector("#account-description").textContent = signedIn
    ? "你的学习数据已保存在私人服务器，并可在登录的设备间同步。"
    : currentAccount?.offline ? "暂时无法连接服务器。学习数据仍保存在本机，连接恢复后可重新登录同步。"
      : setupRequired ? "账号登录需要先启动个人服务器。请在 english-learning 目录运行 python3 server.py，再使用服务器地址打开网站。"
        : "登录后，词库和学习进度会保存到你的个人服务器。";
  accountTabs.classList.toggle("hidden", signedIn || setupRequired);
  accountForm.classList.toggle("hidden", signedIn || setupRequired);
  accountUser.classList.toggle("hidden", !signedIn);
  accountUser.textContent = signedIn ? `已登录为 ${currentAccount.username}${currentAccount.offline ? " · 离线" : ""}` : "";
  document.querySelector("#sign-out").classList.toggle("hidden", !signedIn || currentAccount.offline);
  registerTab.classList.toggle("hidden", !registrationAvailable);
  accountError.classList.add("hidden");
  if (signedIn && currentAccount.offline) {
    document.querySelector("#account-footnote").textContent = "恢复连接后重新登录，可继续同步；当前更改只保存在本机。";
  } else {
    document.querySelector("#account-footnote").textContent = "这是私人自托管账号，不需要邮箱，也不会连接第三方服务。";
  }
}

function setAuthMode(mode) {
  authMode = mode === "register" && registrationAvailable ? "register" : "login";
  document.querySelectorAll("[data-auth-mode]").forEach((button) => button.classList.toggle("active", button.dataset.authMode === authMode));
  const password = document.querySelector('#account-form input[name="password"]');
  const setupField = document.querySelector("#setup-key-field");
  const setupInput = document.querySelector('#account-form input[name="setup_key"]');
  const submit = document.querySelector("#account-submit");
  password.autocomplete = authMode === "register" ? "new-password" : "current-password";
  password.minLength = authMode === "register" ? 12 : 1;
  setupField.classList.toggle("hidden", authMode !== "register");
  setupInput.required = authMode === "register";
  submit.textContent = authMode === "register" ? "创建私人账号" : "登录并同步";
}

function openAccountDialog() {
  updateAccountControls();
  setAuthMode("login");
  document.querySelector("#account-backdrop").classList.remove("hidden");
  if (!currentAccount) document.querySelector('#account-form input[name="username"]').focus();
}

function closeAccountDialog() {
  document.querySelector("#account-backdrop").classList.add("hidden");
}

function saveGuestBackup() {
  try {
    if (!localStorage.getItem(guestBackupKey)) {
      localStorage.setItem(guestBackupKey, localStorage.getItem(storageKey) || JSON.stringify(serializedState()));
    }
  } catch (error) {
    console.warn("Could not keep a local copy before account sign-in.", error);
  }
}

function restoreGuestState() {
  try {
    const backup = localStorage.getItem(guestBackupKey);
    if (backup) localStorage.setItem(storageKey, backup);
    else localStorage.removeItem(storageKey);
    localStorage.removeItem(guestBackupKey);
    localStorage.removeItem(accountMarkerKey);
    Object.assign(state, loadState());
  } catch (error) {
    console.warn("Could not restore the local learning data.", error);
  }
  currentAccount = null;
  renderApplication();
  setSyncStatus("本机保存");
}

async function establishAccountSession(result, username) {
  saveGuestBackup();
  currentAccount = { username, revision: result.revision || 0, offline: false, syncFailed: false };
  try {
    localStorage.setItem(accountMarkerKey, username);
  } catch (error) {
    console.warn("Could not remember the signed-in account on this device.", error);
  }
  applySerializedState(mergeSerializedStates(serializedState(), result.state));
  writeLocalState();
  setSyncStatus("正在保存…", "syncing");
  await flushSync();
  updateAccountControls();
}

async function initializeAccount() {
  if (window.location.protocol === "file:") {
    backendAvailable = false;
    setSyncStatus("服务器不可用", "sync-error");
    updateAccountControls();
    return;
  }
  let rememberedUsername = null;
  try {
    rememberedUsername = localStorage.getItem(accountMarkerKey);
    const session = await requestApi("/api/session");
    backendAvailable = true;
    registrationAvailable = session.registration_open;
    if (session.username) {
      const saved = await requestApi("/api/state");
      if (!rememberedUsername) saveGuestBackup();
      currentAccount = { username: session.username, revision: saved.revision, offline: false, syncFailed: false };
      applySerializedState(mergeSerializedStates(serializedState(), saved.state));
      writeLocalState();
      setSyncStatus("已同步", "synced");
      if (JSON.stringify(mergeSerializedStates(serializedState(), saved.state)) !== JSON.stringify(saved.state)) {
        await flushSync();
      }
    } else {
      if (rememberedUsername) restoreGuestState();
      else setSyncStatus("本机保存");
    }
  } catch (error) {
    if (rememberedUsername) {
      currentAccount = { username: rememberedUsername, revision: null, offline: true };
      setSyncStatus("离线 · 本机保存", "sync-error");
    } else {
      setSyncStatus("服务器不可用", "sync-error");
    }
  }
  updateAccountControls();
}

let syncInProgress = false;
let syncAgain = false;
async function flushSync() {
  if (!currentAccount || currentAccount.offline) return;
  clearTimeout(syncTimer);
  if (syncInProgress) {
    syncAgain = true;
    return;
  }
  syncInProgress = true;
  try {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        const result = await requestApi("/api/state", {
          method: "PUT",
          body: JSON.stringify({ state: serializedState(), revision: currentAccount.revision }),
        });
        currentAccount.revision = result.revision;
        currentAccount.syncFailed = false;
        setSyncStatus("已同步", "synced");
        return;
      } catch (error) {
        if (error.status !== 409 || !error.payload || attempt === 2) throw error;
        currentAccount.revision = error.payload.revision;
        applySerializedState(mergeSerializedStates(serializedState(), error.payload.state));
        writeLocalState();
      }
    }
  } catch (error) {
    if (error.status === 401) currentAccount.offline = true;
    currentAccount.syncFailed = true;
    setSyncStatus(error.status === 401 ? "登录已过期 · 本机保存" : "同步失败 · 本机已保存", "sync-error");
    document.querySelector("#account-error").textContent = error.message;
    document.querySelector("#account-error").classList.remove("hidden");
  } finally {
    syncInProgress = false;
    if (syncAgain) {
      syncAgain = false;
      flushSync();
    }
  }
}

function currentStreak() {
  let date = dateKey();
  if (!state.activeDates.includes(date)) {
    date = dateOffset(date, -1);
    if (!state.activeDates.includes(date)) return 0;
  }
  let streak = 0;
  while (state.activeDates.includes(date)) {
    streak += 1;
    date = dateOffset(date, -1);
  }
  return streak;
}

function recordActivity() {
  const today = dateKey();
  if (!state.activeDates.includes(today)) state.activeDates.push(today);
  const streak = currentStreak();
  document.querySelector("#streak-count").textContent = streak;
  document.querySelector("#streak-stat").textContent = streak;
  renderStreakWeek();
  saveState();
}

function renderStreakWeek() {
  const weekdays = ["一", "二", "三", "四", "五", "六", "日"];
  const monday = dateOffset(dateKey(), -((new Date().getDay() + 6) % 7));
  document.querySelectorAll(".week-dots span").forEach((dot, index) => {
    const active = state.activeDates.includes(dateOffset(monday, index));
    dot.textContent = weekdays[index];
    dot.classList.toggle("complete", active);
    document.querySelectorAll(".streak-track i")[index].classList.toggle("filled", active);
  });
}

function setDate() {
  const now = new Date();
  document.querySelector("#today-date").textContent = now.toLocaleDateString("zh-CN", { month: "long", day: "numeric", weekday: "long" });
  document.querySelector("#writing-date").textContent = now.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" });
  const day = now.toLocaleDateString("en-US", { weekday: "long" }).toUpperCase();
  document.querySelector(".hero-kicker").innerHTML = `<span></span> ${day}, A FRESH START`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function renderWords(filter = "") {
  const normalized = filter.trim().toLowerCase();
  const matching = state.words.filter(({ word, meaning }) => `${word} ${meaning}`.toLowerCase().includes(normalized));
  document.querySelector("#word-list").innerHTML = matching.map((item) => `
    <article class="word-card">
      <div class="word-card-top"><h3>${escapeHtml(item.word)}</h3><button class="delete-word" data-delete="${escapeHtml(item.word)}" aria-label="删除 ${escapeHtml(item.word)}">×</button></div>
      <p class="word-meaning">${escapeHtml(item.meaning)}</p>
      ${item.example ? `<p class="word-example">${escapeHtml(item.example)}</p>` : ""}
    </article>`).join("");
  document.querySelector("#wordbook-count").textContent = state.words.length;
  document.querySelector("#word-total-stat").textContent = state.words.length;
  document.querySelector("#word-added-stat").textContent = Math.max(0, state.words.length - starterWords.length);
  const wordProgress = Math.min(100, Math.round((state.words.length / 50) * 100));
  document.querySelector("#word-progress-fill").style.width = `${wordProgress}%`;
  document.querySelector("#word-progress-percent").textContent = `${wordProgress}%`;
  document.querySelector("#review-nav-count").textContent = Math.min(5, state.words.length);
  document.querySelector("#word-empty").classList.toggle("hidden", matching.length > 0);
  document.querySelector("#wordbook-hint").textContent = state.words.length ? "每个单词，都是一次新的发现。" : "词库空空的，收集你的第一个单词吧。";
  document.querySelector("#preview-words").innerHTML = state.words.slice(-3).reverse().map((item) => `
    <div class="preview-word"><div><strong>${escapeHtml(item.word)}</strong><span>${escapeHtml(item.meaning)}</span></div><em>✳</em></div>`).join("");
  renderReview();
}

function renderReview() {
  const current = reviewQueue[reviewIndex];
  const card = document.querySelector("#flashcard");
  if (!current) {
    document.querySelector("#review-position").textContent = "完成";
    document.querySelector("#review-remaining").textContent = "今天的复习完成了";
    document.querySelector("#review-progress-fill").style.width = "100%";
    document.querySelector("#flash-word").textContent = "Well done!";
    document.querySelector("#flash-phonetic").textContent = "今天的练习";
    document.querySelector("#flash-prompt").textContent = "你已经为今天的自己加了一点分。";
    document.querySelector("#flash-meaning").classList.add("hidden");
    document.querySelector("#flash-example").classList.add("hidden");
    document.querySelector("#review-known").disabled = true;
    document.querySelector("#review-again").disabled = true;
    card.classList.remove("revealed");
    return;
  }
  document.querySelector("#review-known").disabled = false;
  document.querySelector("#review-again").disabled = false;
  document.querySelector("#review-position").textContent = `${String(reviewIndex + 1).padStart(2, "0")} / ${String(reviewQueue.length).padStart(2, "0")}`;
  document.querySelector("#review-remaining").textContent = `还剩 ${reviewQueue.length - reviewIndex} 个`;
  document.querySelector("#review-progress-fill").style.width = `${(reviewIndex / reviewQueue.length) * 100}%`;
  document.querySelector("#flash-word").textContent = current.word;
  document.querySelector("#flash-phonetic").textContent = current.phonetic || "/ˈwɜːrd/";
  document.querySelector("#flash-meaning").textContent = current.meaning;
  document.querySelector("#flash-example").textContent = current.example || "";
  document.querySelector("#flash-prompt").classList.toggle("hidden", isRevealed);
  document.querySelector("#flash-meaning").classList.toggle("hidden", !isRevealed);
  document.querySelector("#flash-example").classList.toggle("hidden", !isRevealed || !current.example);
  card.classList.toggle("revealed", isRevealed);
}

function showPage(page) {
  document.querySelectorAll(".page").forEach((section) => section.classList.toggle("active", section.id === `page-${page}`));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.page === page));
  const currentNav = document.querySelector(`.nav-item[data-page="${page}"] span:nth-child(2)`);
  document.querySelector("#page-crumb").textContent = currentNav.textContent;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderTasks() {
  const checkboxes = [...document.querySelectorAll(".task-item input")];
  checkboxes.forEach((input) => { input.checked = Boolean(state.tasks[input.dataset.task]); });
  const complete = checkboxes.filter((input) => input.checked).length;
  document.querySelector("#task-progress").textContent = `${complete} / ${checkboxes.length} 完成`;
  document.querySelector("#tasks-done-stat").textContent = complete;
  document.querySelector("#task-progress-fill").style.width = `${(complete / checkboxes.length) * 100}%`;
  document.querySelector("#task-progress-percent").textContent = `${Math.round((complete / checkboxes.length) * 100)}%`;
}

function renderCourse() {
  const completed = new Set(state.completedLessons);
  const selectedCourse = state.selectedCourse === "foundation" || resourceTracks.some((track) => track.id === state.selectedCourse)
    ? state.selectedCourse : "foundation";
  state.selectedCourse = selectedCourse;
  const currentTrack = resourceTracks.find((track) => track.id === selectedCourse);
  const lessons = selectedCourse === "foundation"
    ? courseWeeks.flatMap((week, weekIndex) => week.lessons.map((lesson, dayIndex) => ({
      id: `w${weekIndex + 1}d${dayIndex + 1}`, title: lesson[0], weekIndex, dayIndex,
    })))
    : currentTrack.units.map(([title], index) => ({ id: `${selectedCourse}-u${index + 1}`, title, index }));
  const lessonCount = lessons.length;
  const finishedCount = lessons.filter((lesson) => completed.has(lesson.id)).length;
  const nextLesson = lessons.find((lesson) => !completed.has(lesson.id));
  if (selectedCourse === "foundation" && nextLesson) openCourseWeek = nextLesson.weekIndex;

  document.querySelector("#course-completed").textContent = finishedCount;
  document.querySelector("#course-progress-fill").style.width = `${(finishedCount / lessonCount) * 100}%`;
  document.querySelector("#course-progress-caption").textContent = finishedCount === lessonCount
    ? "太棒了！你已经完成这条学习路线。"
    : finishedCount
      ? selectedCourse === "foundation"
        ? `第 ${nextLesson.weekIndex + 1} 周 · ${nextLesson.title}`
        : `下一单元 · ${nextLesson.title}`
      : "刚刚起步，正好从第一课开始。";
  document.querySelector("#continue-course").textContent = nextLesson
    ? selectedCourse === "foundation"
      ? `继续第 ${nextLesson.weekIndex + 1} 周 · 第 ${nextLesson.dayIndex + 1} 课 →`
      : `继续：${nextLesson.title} →`
    : "完成全部课程 ✓";
  document.querySelector("#continue-course").disabled = !nextLesson;
  document.querySelector("#active-course-kicker").textContent = selectedCourse === "foundation" ? "YOUR STEP-BY-STEP PLAN" : "ORIGINAL STUDY GUIDE";
  document.querySelector("#active-course-heading").textContent = selectedCourse === "foundation" ? "12 周零基础学习路线" : currentTrack.title;
  document.querySelector("#active-course-caption").textContent = selectedCourse === "foundation"
    ? "每周 5 课 · 周末休息或轻松复习"
    : `${currentTrack.duration} · 完成 ${finishedCount} / ${lessonCount} 单元`;

  document.querySelector("#course-library").innerHTML = [
    {
      id: "foundation", title: "零基础起步", level: "字母到日常对话", duration: "12 周 · 60 课",
      done: state.completedLessons.filter((id) => /^w\d+d\d+$/.test(id)).length,
      total: courseWeeks.reduce((total, week) => total + week.lessons.length, 0),
    },
    ...resourceTracks.map((track) => ({
      ...track,
      done: track.units.filter((_, index) => completed.has(`${track.id}-u${index + 1}`)).length,
      total: track.units.length,
    })),
  ].map((track) => `<button class="course-choice ${selectedCourse === track.id ? "selected" : ""}" data-course="${track.id}" aria-pressed="${selectedCourse === track.id}">
    <span class="course-choice-level">${track.level}</span><strong>${track.title}</strong><span class="course-choice-duration">${track.duration}</span>
    <span class="course-choice-description">${track.description}</span><span class="course-choice-progress">${track.done} / ${track.total} 完成</span>
  </button>`).join("");

  if (currentTrack) {
    const currentIndex = lessons.findIndex((lesson) => !completed.has(lesson.id));
    document.querySelector("#course-phases").innerHTML = `<section class="course-phase course-resource-units"><h3>${currentTrack.title} · 原创学习任务</h3>${currentTrack.units.map(([title, focus, practice], index) => {
      const lessonId = `${selectedCourse}-u${index + 1}`;
      const isDone = completed.has(lessonId);
      const isNext = index === currentIndex;
      return `<article class="course-lesson course-unit ${isDone ? "lesson-done" : ""} ${isNext ? "lesson-current" : ""}" id="course-unit-${lessonId}">
        <span class="lesson-day">UNIT ${String(index + 1).padStart(2, "0")}</span>
        <div class="lesson-content"><strong>${title}</strong><p><b>学习重点：</b>${focus}</p><span class="lesson-example">${practice}</span></div>
        <button class="lesson-toggle" data-lesson="${lessonId}" aria-pressed="${isDone}">${isDone ? "✓ 已完成" : "标记完成"}</button>
      </article>`;
    }).join("")}</section>`;
    return;
  }

  const phases = [];
  courseWeeks.forEach((week, weekIndex) => {
    if (!phases.includes(week.phase)) phases.push(week.phase);
  });
  document.querySelector("#course-phases").innerHTML = phases.map((phase) => {
    const weeks = courseWeeks.map((week, weekIndex) => ({ ...week, weekIndex })).filter((week) => week.phase === phase);
    return `<section class="course-phase"><h3>${phase}</h3>${weeks.map((week) => {
      const weekDone = week.lessons.filter((_, dayIndex) => completed.has(`w${week.weekIndex + 1}d${dayIndex + 1}`)).length;
      const isCurrent = nextLesson && nextLesson.weekIndex === week.weekIndex;
      return `<details class="course-week" id="course-week-${week.weekIndex}" ${week.weekIndex === openCourseWeek ? "open" : ""}>
        <summary>
          <span class="week-number">${String(week.weekIndex + 1).padStart(2, "0")}</span>
          <span class="week-summary-copy"><strong>${week.title}</strong><small>${week.goal}</small></span>
          <span class="week-status">${weekDone === week.lessons.length ? "已完成" : isCurrent ? "正在学习" : `${weekDone} / 5 课`}</span>
          <span class="week-chevron">⌄</span>
        </summary>
        <div class="course-lessons">${week.lessons.map(([title, instruction, example], dayIndex) => {
          const lessonId = `w${week.weekIndex + 1}d${dayIndex + 1}`;
          const isDone = completed.has(lessonId);
          const isNext = nextLesson?.weekIndex === week.weekIndex && nextLesson?.dayIndex === dayIndex;
          return `<article class="course-lesson ${isDone ? "lesson-done" : ""} ${isNext ? "lesson-current" : ""}">
            <span class="lesson-day">DAY ${dayIndex + 1}</span>
            <div class="lesson-content"><strong>${title}</strong><p>${instruction}</p><span class="lesson-example">${example}</span></div>
            <button class="lesson-toggle" data-lesson="${lessonId}" aria-pressed="${isDone}">${isDone ? "✓ 已完成" : "标记完成"}</button>
          </article>`;
        }).join("")}</div>
      </details>`;
    }).join("")}</section>`;
  }).join("");
}

function updateReviewStats() {
  document.querySelector("#reviewed-count").textContent = reviewedToday;
  document.querySelector("#aside-review-fill").style.width = `${Math.min(100, reviewedToday * 20)}%`;
}

document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", () => showPage(item.dataset.page)));
document.querySelectorAll(".account-trigger").forEach((button) => button.addEventListener("click", openAccountDialog));
document.querySelector("#close-account").addEventListener("click", closeAccountDialog);
document.querySelector("#account-backdrop").addEventListener("click", (event) => {
  if (event.target === event.currentTarget) closeAccountDialog();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeAccountDialog();
});
document.querySelectorAll("[data-auth-mode]").forEach((button) => button.addEventListener("click", () => setAuthMode(button.dataset.authMode)));
document.querySelector("#account-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const values = new FormData(form);
  const payload = {
    username: String(values.get("username")).trim(),
    password: String(values.get("password")),
  };
  const endpoint = authMode === "register" ? "/api/register" : "/api/login";
  if (authMode === "register") payload.setup_key = String(values.get("setup_key")).trim();
  const submit = document.querySelector("#account-submit");
  const errorBox = document.querySelector("#account-error");
  submit.disabled = true;
  errorBox.classList.add("hidden");
  try {
    const result = await requestApi(endpoint, { method: "POST", body: JSON.stringify(payload) });
    registrationAvailable = false;
    await establishAccountSession(result, result.username);
    form.reset();
    closeAccountDialog();
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.classList.remove("hidden");
  } finally {
    submit.disabled = false;
  }
});
document.querySelector("#sign-out").addEventListener("click", async () => {
  const button = document.querySelector("#sign-out");
  button.disabled = true;
  try {
    clearTimeout(syncTimer);
    await flushSync();
    while (syncInProgress) await new Promise((resolve) => setTimeout(resolve, 50));
    if (currentAccount?.syncFailed) throw new Error("有未同步的修改。请恢复连接并完成同步后再退出，以免丢失这些修改。");
    await requestApi("/api/logout", { method: "POST", body: "{}" });
    restoreGuestState();
    updateAccountControls();
    closeAccountDialog();
  } catch (error) {
    const errorBox = document.querySelector("#account-error");
    errorBox.textContent = error.message;
    errorBox.classList.remove("hidden");
  } finally {
    button.disabled = false;
  }
});
document.querySelectorAll("[data-go]").forEach((button) => button.addEventListener("click", () => {
  showPage(button.dataset.go);
  if (button.dataset.go === "words") document.querySelector("#add-word-form").classList.remove("hidden");
}));
document.querySelector("#continue-course").addEventListener("click", () => {
  showPage("course");
  if (state.selectedCourse === "foundation") {
    const week = document.querySelector(`#course-week-${openCourseWeek}`);
    if (week) {
      week.open = true;
      week.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  } else {
    const firstIncomplete = resourceTracks.find((track) => track.id === state.selectedCourse)?.units.findIndex((_, index) =>
      !state.completedLessons.includes(`${state.selectedCourse}-u${index + 1}`));
    if (firstIncomplete >= 0) {
      document.querySelector(`#course-unit-${state.selectedCourse}-u${firstIncomplete}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }
});
document.querySelector("#course-library").addEventListener("click", (event) => {
  const choice = event.target.closest("[data-course]");
  if (!choice) return;
  state.selectedCourse = choice.dataset.course;
  openCourseWeek = 0;
  saveState();
  renderCourse();
});
document.querySelector("#course-phases").addEventListener("click", (event) => {
  const button = event.target.closest("[data-lesson]");
  if (!button) return;
  const lessonId = button.dataset.lesson;
  const completed = new Set(state.completedLessons);
  const wasCompleted = completed.has(lessonId);
  if (wasCompleted) {
    completed.delete(lessonId);
  } else {
    completed.add(lessonId);
  }
  state.completedLessons = [...completed];
  if (wasCompleted) saveState();
  else recordActivity();
  renderCourse();
  const selectedTrack = resourceTracks.find((track) => track.id === state.selectedCourse);
  if (selectedTrack) {
    const nextIndex = selectedTrack.units.findIndex((_, index) => !state.completedLessons.includes(`${selectedTrack.id}-u${index + 1}`));
    if (nextIndex >= 0) document.querySelector(`#course-unit-${selectedTrack.id}-u${nextIndex}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});

document.querySelectorAll(".task-item input").forEach((input) => input.addEventListener("change", () => {
  state.tasks[input.dataset.task] = input.checked;
  if (input.checked) recordActivity();
  saveState();
  renderTasks();
}));

document.querySelector("#open-word-form").addEventListener("click", () => {
  document.querySelector("#add-word-form").classList.toggle("hidden");
  document.querySelector('#add-word-form input[name="word"]').focus();
});
document.querySelector("#cancel-word-form").addEventListener("click", () => {
  document.querySelector("#add-word-form").reset();
  document.querySelector("#add-word-form").classList.add("hidden");
});
document.querySelector("#add-word-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const word = String(form.get("word")).trim();
  if (state.words.some((item) => item.word.toLowerCase() === word.toLowerCase())) {
    document.querySelector('#add-word-form input[name="word"]').setCustomValidity("这个单词已经在词库里了。");
    document.querySelector('#add-word-form input[name="word"]').reportValidity();
    return;
  }
  state.words.push({
    word,
    meaning: String(form.get("meaning")).trim(),
    example: String(form.get("example")).trim(),
    phonetic: "/ˈwɜːrd/",
  });
  recordActivity();
  reviewQueue = state.words.slice(0, 5);
  reviewIndex = 0;
  isRevealed = false;
  retriedWords.clear();
  state.tasks = { ...state.tasks, review: false };
  saveState();
  event.currentTarget.reset();
  event.currentTarget.classList.add("hidden");
  renderWords(document.querySelector("#word-search").value);
  renderTasks();
});
document.querySelector('#add-word-form input[name="word"]').addEventListener("input", (event) => event.currentTarget.setCustomValidity(""));
document.querySelector("#word-search").addEventListener("input", (event) => renderWords(event.currentTarget.value));
document.querySelector("#word-list").addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete]");
  if (!button) return;
  state.words = state.words.filter((item) => item.word !== button.dataset.delete);
  reviewQueue = state.words.slice(0, 5);
  reviewIndex = 0;
  isRevealed = false;
  retriedWords.clear();
  saveState();
  renderWords(document.querySelector("#word-search").value);
});
document.querySelector("#flashcard").addEventListener("click", () => {
  if (!reviewQueue[reviewIndex]) return;
  isRevealed = !isRevealed;
  renderReview();
});
document.querySelector("#review-known").addEventListener("click", () => {
  if (!reviewQueue[reviewIndex]) return;
  reviewedToday += 1;
  recordActivity();
  reviewIndex += 1;
  isRevealed = false;
  updateReviewStats();
  renderReview();
});
document.querySelector("#review-again").addEventListener("click", () => {
  if (!reviewQueue[reviewIndex]) return;
  const current = reviewQueue[reviewIndex];
  if (!retriedWords.has(current.word)) {
    reviewQueue.push(current);
    retriedWords.add(current.word);
  }
  reviewIndex += 1;
  isRevealed = false;
  renderReview();
});

const writingInput = document.querySelector("#writing-input");
writingInput.value = state.writing;
function updateWritingCount() {
  const count = writingInput.value.trim().split(/\s+/).filter(Boolean).length;
  document.querySelector("#writing-words").textContent = count;
}
writingInput.addEventListener("input", () => {
  state.writing = writingInput.value;
  if (writingInput.value.trim()) recordActivity();
  saveState();
  updateWritingCount();
});
document.querySelectorAll(".phrase").forEach((button) => button.addEventListener("click", () => {
  const spacer = writingInput.value && !/\s$/.test(writingInput.value) ? " " : "";
  writingInput.value += `${spacer}${button.dataset.phrase} `;
  writingInput.dispatchEvent(new Event("input"));
  writingInput.focus();
}));

setDate();
const streak = currentStreak();
document.querySelector("#streak-count").textContent = streak;
document.querySelector("#streak-stat").textContent = streak;
renderStreakWeek();
renderWords();
renderTasks();
renderCourse();
updateReviewStats();
updateWritingCount();
let hasRememberedAccount = false;
try {
  hasRememberedAccount = Boolean(localStorage.getItem(accountMarkerKey));
} catch (error) {
  console.warn("Could not read the saved account marker.", error);
}
setSyncStatus(hasRememberedAccount ? "检查账号…" : "本机保存");
initializeAccount();
