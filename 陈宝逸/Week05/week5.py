import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random

# -------------------- 1. 超参数设置 --------------------
batch_size = 64
initial_block_size = 128          # 期望的上下文长度，若数据不足会自动缩小
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# -------------------- 2. 数据准备（字符级，使用示例文本）--------------------
text = """
N THE 24th of February, 1810, the look-out at Notre-Dame de la Garde signalled the three-master, the Pharaon from Smyrna, Trieste, and Naples.
As usual, a pilot put off immediately, and rounding the Chateau d'If, got on board the vessel between Cape Morgion and Rion island.
Immediately, and according to custom, the ramparts of Fort Saint-Jean were covered with spectators; it is always an event at Marseilles for a ship to come into port, especially when this ship, like the Pharaon, has been built, rigged, and laden at the old Phocée docks, and belongs to an owner of the city.
The ship drew on and had safely passed the strait, which some volcanic shock has made between the Calasareigne and Jaros islands; had doubled Pomègue, and approached the harbor under topsails, jib, and spanker, but so slowly and sedately that the idlers, with that instinct which is the forerunner of evil, asked one another what misfortune could have happened on board. However, those experienced in navigation saw plainly that if any accident had occurred, it was not to the vessel herself, for she bore down with all the evidence of being skilfully handled, the anchor a-cockbill, the jib-boom guys already eased off, and standing by the side of the pilot, who was steering the Pharaon towards the narrow entrance of the inner port, was a young man, who, with activity and vigilant eye, watched every motion of the ship, and repeated each direction of the pilot.
The vague disquietude which prevailed among the spectators had so much affected one of the crowd that he did not await the arrival of the vessel in harbor, but jumping into a small skiff, desired to be pulled alongside the Pharaon, which he reached as she rounded into La Rèserve basin.
When the young man on board saw this person approach, he left his station by the pilot, and, hat in hand, leaned over the ship's bulwarks.
He was a fine, tall, slim young fellow of eighteen or twenty, with black eyes, and hair as dark as a raven's wing; and his whole appearance bespoke that calmness and resolution peculiar to men accustomed from their cradle to contend with danger.
Ah, is it you, Dantès?" cried the man in the skiff. "What's the matter? and why have you such an air of sadness aboard
A great misfortune, M. Morrel," replied the young man,--"a great misfortune, for me especially! Off Civita Vecchia we lost our brave Captain Leclere.
And the cargo?" inquired the owner, eagerly.
Is all safe, M. Morrel; and I think you will be satisfied on that head. But poor Captain Leclere
What happened to him?" asked the owner, with an air of considerable resignation. "What happened to the worthy captain
He died.
Fell into the sea
No, sir, he died of brain-fever in dreadful agony." Then turning to the crew, he said, "Bear a hand there, to take in sail
All hands obeyed, and at once the eight or ten seamen who composed the crew, sprang to their respective stations at the spanker brails and outhaul, topsail sheets and halyards, the jib downhaul, and the topsail clewlines and buntlines. The young sailor gave a look to see that his orders were promptly and accurately obeyed, and then turned again to the owner.
And how did this misfortune occur?" inquired the latter, resuming the interrupted conversation.
Alas, sir, in the most unexpected manner. After a long talk with the harbor-master, Captain Leclere left Naples greatly disturbed in mind. In twenty-four hours he was attacked by a fever, and died three days afterwards. We performed the usual burial service, and he is at his rest, sewn up in his hammock with a thirty-six pound shot at his head and his heels, off El Giglio island. We bring to his widow his sword and cross of honor. It was worth while, truly," added the young man with a melancholy smile, "to make war against the English for ten years, and to die in his bed at last, like everybody else.
Why, you see, Edmond," replied the owner, who appeared more comforted at every moment, "we are all mortal, and the old must make way for the young. If not, why, there would be no promotion; and since you assure me that the cargo
Is all safe and sound, M. Morrel, take my word for it; and I advise you not to take 25,000 francs for the profits of the voyage.
Then, as they were just passing the Round Tower, the young man shouted: "Stand by there to lower the topsails and jib; brail up the spanker
The order was executed as promptly as it would have been on board a man-of-war.
Let go--and clue up!" At this last command all the sails were lowered, and the vessel moved almost imperceptibly onwards.
Now, if you will come on board, M. Morrel," said Dantès, observing the owner's impatience, "here is your supercargo, M. Danglars, coming out of his cabin, who will furnish you with every particular. As for me, I must look after the anchoring, and dress the ship in mourning.
The owner did not wait for a second invitation. He seized a rope which Dantès flung to him, and with an activity that would have done credit to a sailor, climbed up the side of the ship, while the young man, going to his task, left the conversation to Danglars, who now came towards the owner. He was a man of twenty-five or twenty-six years of age, of unprepossessing countenance, obsequious to his superiors, insolent to his subordinates; and this, in addition to his position as responsible agent on board, which is always obnoxious to the sailors, made him as much disliked by the crew as Edmond Dantès was beloved by them.
Well, M. Morrel," said Danglars, "you have heard of the misfortune that has befallen us
Yes--yes: poor Captain Leclere! He was a brave and an honest man.
And a first-rate seaman, one who had seen long and honorable service, as became a man charged with the interests of a house so important as that of Morrel & Son," replied Danglars.
But," replied the owner, glancing after Dantès, who was watching the anchoring of his vessel, "it seems to me that a sailor needs not be so old as you say, Danglars, to understand his business, for our friend Edmond seems to understand it thoroughly, and not to require instruction from any one.
Yes," said Danglars, darting at Edmond a look gleaming with hate. "Yes, he is young, and youth is invariably self-confident. Scarcely was the captain's breath out of his body when he assumed the command without consulting any one, and he caused us to lose a day and a half at the Island of Elba, instead of making for Marseilles direct.
As to taking command of the vessel," replied Morrel, "that was his duty as captain's mate; as to losing a day and a half off the Island of Elba, he was wrong, unless the vessel needed repairs.
The vessel was in as good condition as I am, and as, I hope you are, M. Morrel, and this day and a half was lost from pure whim, for the pleasure of going ashore, and nothing else.
Dantès," said the shipowner, turning towards the young man, "come this way
In a moment, sir," answered Dantès, "and I'm with you." Then calling to the crew, he said--"Let go
The anchor was instantly dropped, and the chain ran rattling through the port-hole. Dantès continued at his post in spite of the presence of the pilot, until this manoeuvre was completed, and then he added, "Half-mast the colors, and square the yards
You see," said Danglars, "he fancies himself captain already, upon my word.
And so, in fact, he is," said the owner.
Except your signature and your partner's, M. Morrel.
And why should he not have this?" asked the owner; "he is young, it is true, but he seems to me a thorough seaman, and of full experience.
A cloud passed over Danglars' brow. "Your pardon, M. Morrel," said Dantès, approaching, "the vessel now rides at anchor, and I am at your service. You hailed me, I think
Danglars retreated a step or two.
I wished to inquire why you stopped at the Island of Elba
I do not know, sir; it was to fulfil the last instructions of Captain Leclere, who, when dying, gave me a packet for Marshal Bertrand.
Then did you see him, Edmond
Who
The marshal.
Yes.
Morrel looked around him, and then, drawing Dantès on one side, he said suddenly--"And how is the emperor
Very well, as far as I could judge from the sight of him.
You saw the emperor, then
He entered the marshal's apartment while I was there.
And you spoke to him
Why, it was he who spoke to me, sir," said Dantès, with a smile.
And what did he say to you
Asked me questions about the vessel, the time she left Marseilles, the course she had taken, and what was her cargo. I believe, if she had not been laden, and I had been her master, he would have bought her. But I told him I was only mate, and that she belonged to the firm of Morrel & Son. 'Ah, yes,' he said, 'I know them. The Morrels have been shipowners from father to son; and there was a Morrel who served in the same regiment with me when I was in garrison at Valence.
Pardieu! and that is true!" cried the owner, greatly delighted. "And that was Policar Morrel, my uncle, who was afterwards a captain. Dantès, you must tell my uncle that the emperor remembered him, and you will see it will bring tears into the old soldier's eyes. Come, come," continued he, patting Edmond's shoulder kindly, "you did very right, Dantès, to follow Captain Leclere's instructions, and touch at Elba, although if it were known that you had conveyed a packet to the marshal, and had conversed with the emperor, it might bring you into trouble.
How could that bring me into trouble, sir?" asked Dantès; "for I did not even know of what I was the bearer; and the emperor merely made such inquiries as he would of the first comer. But, pardon me, here are the health officers and the customs inspectors coming alongside." And the young man went to the gangway. As he departed, Danglars approached, and said
Well, it appears that he has given you satisfactory reasons for his landing at Porto-Ferrajo
Yes, most satisfactory, my dear Danglars.
Well, so much the better," said the supercargo; "for it is not pleasant to think that a comrade has not done his duty.
Dantès has done his," replied the owner, "and that is not saying much. It was Captain Leclere who gave orders for this delay.
Talking of Captain Leclere, has not Dantès given you a letter from him
To me?--no--was there one
I believe that, besides the packet, Captain Leclere confided a letter to his care.
Of what packet are you speaking, Danglars
Why, that which Dantès left at Porto-Ferrajo.
How do you know he had a packet to leave at Porto-Ferrajo
Danglars turned very red.
I was passing close to the door of the captain's cabin, which was half open, and I saw him give the packet and letter to Dantès.
He did not speak to me of it," replied the shipowner; "but if there be any letter he will give it to me.
Danglars reflected for a moment. "Then, M. Morrel, I beg of you," said he, "not to say a word to Dantès on the subject. I may have been mistaken.
At this moment the young man returned; Danglars withdrew.
Well, my dear Dantès, are you now free?" inquired the owner.
Yes, sir.
You have not been long detained.
No. I gave the custom-house officers a copy of our bill of lading; and as to the other papers, they sent a man off with the pilot, to whom I gave them.
Then you have nothing more to do here
No--everything is all right now.
Then you can come and dine with me
I really must ask you to excuse me, M. Morrel. My first visit is due to my father, though I am not the less grateful for the honor you have done me.
Right, Dantès, quite right. I always knew you were a good son.
And," inquired Dantès, with some hesitation, "do you know how my father is
Well, I believe, my dear Edmond, though I have not seen him lately.
Yes, he likes to keep himself shut up in his little room.
That proves, at least, that he has wanted for nothing during your absence.
Dantès smiled. "My father is proud, sir, and if he had not a meal left, I doubt if he would have asked anything from anyone, except from Heaven.
Well, then, after this first visit has been made we shall count on you.
I must again excuse myself, M. Morrel, for after this first visit has been paid I have another which I am most anxious to pay." "True, Dantès, I forgot that there was at the Catalans some one who expects you no less impatiently than your father--the lovely Mercédès.
Dantès blushed.
Ah, ha," said the shipowner, "I am not in the least surprised, for she has been to me three times, inquiring if there were any news of the Pharaon. Peste! Edmond, you have a very handsome mistress
She is not my mistress," replied the young sailor, gravely; "she is my betrothed.
Sometimes one and the same thing," said Morrel, with a smile.
Not with us, sir," replied Dantès.
Well, well, my dear Edmond," continued the owner, "don't let me detain you. You have managed my affairs so well that I ought to allow you all the time you require for your own. Do you want any money
No, sir; I have all my pay to take--nearly three months' wages.
You are a careful fellow, Edmond.
Say I have a poor father, sir.
Yes, yes, I know how good a son you are, so now hasten away to see your father. I have a son too, and I should be very wroth with those who detained him from me after a three months' voyage.
Then I have your leave, sir
Yes, if you have nothing more to say to me.
Nothing.
Captain Leclere did not, before he died, give you a letter for me
He was unable to write, sir. But that reminds me that I must ask your leave of absence for some days.
To get married
Yes, first, and then to go to Paris.
Very good; have what time you require, Dantès. It will take quite six weeks to unload the cargo, and we cannot get you ready for sea until three months after that; only be back again in three months, for the Pharaon," added the owner, patting the young sailor on the back, "cannot sail without her captain.
Without her captain!" cried Dantès, his eyes sparkling with animation; "pray mind what you say, for you are touching on the most secret wishes of my heart. Is it really your intention to make me captain of the Pharaon
If I were sole owner we'd shake hands on it now, my dear Dantès, and call it settled; but I have a partner, and you know the Italian proverb--Chi ha compagno ha padrone--'He who has a partner has a master.' But the thing is at least half done, as you have one out of two votes. Rely on me to procure you the other; I will do my best.
Ah, M. Morrel," exclaimed the young seaman, with tears in his eyes, and grasping the owner's hand, "M. Morrel, I thank you in the name of my father and of Mercédès.
That's all right, Edmond. There's a providence that watches over the deserving. Go to your father: go and see Mercédès, and afterwards come to me.
Shall I row you ashore
No, thank you; I shall remain and look over the accounts with Danglars. Have you been satisfied with him this voyage
That is according to the sense you attach to the question, sir. Do you mean is he a good comrade? No, for I think he never liked me since the day when I was silly enough, after a little quarrel we had, to propose to him to stop for ten minutes at the island of Monte Cristo to settle the dispute--a proposition which I was wrong to suggest, and he quite right to refuse. If you mean as responsible agent when you ask me the question, I believe there is nothing to say against him, and that you will be content with the way in which he has performed his duty.
But tell me, Dantès, if you had command of the Pharaon should you be glad to see Danglars remain
Captain or mate, M. Morrel, I shall always have the greatest respect for those who possess the owners' confidence.
That's right, that's right, Dantès! I see you are a thoroughly good fellow, and will detain you no longer. Go, for I see how impatient you are.
Then I have leave
Go, I tell you.
May I have the use of your skiff
Certainly.
Then, for the present, M. Morrel, farewell, and a thousand thanks
I hope soon to see you again, my dear Edmond. Good luck to you.
The young sailor jumped into the skiff, and sat down in the stern sheets, with the order that he be put ashore at La Canebière. The two oarsmen bent to their work, and the little boat glided away as rapidly as possible in the midst of the thousand vessels which choke up the narrow way which leads between the two rows of ships from the mouth of the harbor to the Quai d'Orleans.
The shipowner, smiling, followed him with his eyes until he saw him spring out on the quay and disappear in the midst of the throng, which from five o'clock in the morning until nine o'clock at night, swarms in the famous street of La Canebière,--a street of which the modern Phocaeans are so proud that they say with all the gravity in the world, and with that accent which gives so much character to what is said, "If Paris had La Canebière, Paris would be a second Marseilles." On turning round the owner saw Danglars behind him, apparently awaiting orders, but in reality also watching the young sailor,--but there was a great difference in the expression of the two men who thus followed the movements of Edmond Dantès.
"""

# 构建字符级 token 映射
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 编码整个文本
data = torch.tensor(encode(text), dtype=torch.long)

# 调整 block_size 以适应数据长度
if len(data) <= initial_block_size + 1:
    block_size = max(1, len(data) // 2)
    print(f"警告：数据长度 ({len(data)}) 小于初始 block_size ({initial_block_size})，已自动调整为 {block_size}")
else:
    block_size = initial_block_size

# 划分训练/验证集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 确保 len(data) - block_size 至少为 1
    max_start = len(data) - block_size
    if max_start <= 0:
        # 数据仍不足，返回空 batch（实际训练时应避免）
        raise ValueError(f"数据长度 {len(data)} 不足，无法生成长度为 {block_size} 的序列。请增加文本或减小 block_size。")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# -------------------- 3. 模型定义（使用动态 block_size）--------------------
class Head(nn.Module):
    """单头注意力"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= block_size, f"序列长度 {T} 超过 block_size {block_size}"
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel().to(device)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"实际使用的 block_size = {block_size}")

# -------------------- 4. 训练 --------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                X, Y = get_batch(split)
            except ValueError as e:
                print(e)
                break
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -------------------- 5. 文本生成 --------------------
context = "It was the best of times"
context_ids = encode(context)
x = torch.tensor([context_ids], dtype=torch.long, device=device)
generated_ids = model.generate(x, max_new_tokens=200, temperature=0.8, top_k=40)
generated_text = decode(generated_ids[0].tolist())
print("\n生成的文本：\n", generated_text)

# -------------------- 6. 模型保存 --------------------

# 保存模型和关键参数
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'block_size': block_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout,
    'stoi': stoi,      # 字符到索引映射（可选）
    'itos': itos       # 索引到字符映射（可选）
}
torch.save(checkpoint, 'language_model_checkpoint.pth')
print("模型及配置已保存为 language_model_checkpoint.pth")
