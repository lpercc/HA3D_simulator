"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from param import args

logger = logging.getLogger(__name__)

import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)
    
class CrossModalityAttention(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, hidden_dim):
        super(CrossModalityAttention, self).__init__()
        self.image_query = nn.Linear(image_feature_dim, hidden_dim)
        self.text_key = nn.Linear(text_feature_dim, hidden_dim)
        self.text_value = nn.Linear(text_feature_dim, hidden_dim)
        self.sqrt_dk = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float))

    def forward(self, image_features, text_features):
        # image_features: (batch_size, image_feature_dim)
        # text_features: (batch_size, text_feature_dim)
        
        # Generate query from image features
        query = self.image_query(image_features)  # (batch_size, hidden_dim)
        
        # Generate key and value from text features
        key = self.text_key(text_features)        # (batch_size, hidden_dim)
        value = self.text_value(text_features)    # (batch_size, hidden_dim)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.sqrt_dk
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to the values
        attended_text = torch.matmul(attention_weights, value)
        
        return attended_text
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768 
    max_length = 5 #NOTE - This is context length
    whole_step = 30 #TODO - Make its related to the max_eposide_length

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.time_emb = nn.Embedding(config.whole_step, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # our task, state 
        if args.feature_type == 'fused': # set fused means the image feature only with batch_size, 2048 from resnet 152
            if args.fusion_type == 'simple':
                self.state_encoder = nn.Sequential(nn.Linear(2048+768, config.n_embd), nn.Tanh())
            elif args.fusion_type == 'attention':
                image_feature_dim = 2048
                text_feature_dim = 768
                self.state_encoder = CrossModalityAttention(image_feature_dim, text_feature_dim, config.n_embd)
                #REVIEW - self.proj = nn.Linear(config.n_embd * 2, config.n_embd)
            elif args.fusion_type == 'bert':
                image_feature_dim = 2048
                text_feature_dim = 768
                #self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embd))
                self.image_embedding = nn.Linear(image_feature_dim, config.n_embd)
                #REVIEW - Need position embedding? 
                self.image_pos_emb = nn.Parameter(torch.randn(1, config.n_embd)) 
                self.text_embedding = nn.Identity()
                self.text_pos_emb = nn.Parameter(torch.randn(1, config.n_embd))
                encoder_layer = nn.TransformerEncoderLayer(d_model=config.n_embd * 2, nhead=2, batch_first=True)
                self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.bert_layers)
                self.proj = nn.Linear(config.n_embd * 2, config.n_embd)
            else: 
                raise NotImplementedError('Fusion type not implemented')
        elif args.feature_type == 'feature_map':
            self.state_encoder = nn.Sequential(nn.Linear())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, \
            torch.nn.TransformerEncoder, torch.nn.LayerNorm, nn.TransformerEncoderLayer)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif 'image_pos_emb' in fpn or 'text_pos_emb' in fpn:
                    no_decay.add(fpn)


        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')
        # no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # TODO: The original code is wrong, need to change it to the correct version
        # Here we do not care the input length, we only care the context length and it's initial sequence modeling task 
        # In our setting, the block_size should be 15
        # Final context length should be block_size
        # states: (batch, block_size // 3, 2048 + 768) 
        # actions: (batch, block_size // 3 , 1)
        # targets: (batch, block_size // 3, 1)
        # rtgs: (batch, block_size // 3, 1)
        # timesteps: (batch, block_size // 3  , 1) , the shape of timsteps should be the same as states

        
        assert self.model_type == 'reward_conditioned', "The model type should be reward_conditioned"
        if args.fusion_type == 'simple':
            state_embeddings = self.state_encoder(states.type(torch.float32)) # (batch , block_size // 3, n_embd)
        elif args.fusion_type == 'attention': 
            image_features = states[:, :, :2048]
            text_features = states[:, :, 2048:]
            image_embeddings = F.normalize(self.image_embedding(image_features), dim=-1)
            text_embeddings = F.normolize(self.text_embedding(text_features), dim=-1)
            #NOTE - Here we use text as query
            state_embeddings = self.state_encoder(image_embeddings, text_embeddings, text_embeddings)
        elif args.fusion_type == 'bert':
            image_features = states[:, :, :2048]
            text_features = states[:, :, 2048:]
            #NOTE - sure the text_image_embedding is masked
            text_embedding = F.normalize(self.text_embedding(text_features), dim=-1) + self.text_pos_emb
            image_embedding = F.normalize(self.image_embedding(image_features), dim=-1) + self.image_pos_emb
            text_image_embedding = torch.cat([text_embedding, image_embedding], dim=1) # cat on the sequence Length dim
            state_embeddings = self.state_encoder(text_image_embedding)
            
            
        
        rtg_embeddings = self.ret_emb(rtgs.type(torch.float32)) # (batch, block_size // 3, n_embd)
        
        if actions is not None and self.model_type == 'reward_conditioned':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size // 3, n_embd)
        elif actions is None and self.model_type == 'reward_conditioned':
            action_embeddings = torch.zeros((states.shape[0], states.shape[1], self.config.n_embd), dtype=torch.float32, device=state_embeddings.device) # (batch, block_size // 3, n_embd)
            
        time_embeddings = self.time_emb(timesteps.type(torch.long).squeeze(-1)) # (batch, block_size // 3, n_embd)
            
        # add together the embeddings
        state_embeddings = state_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        
        token_embeddings = torch.stack([rtg_embeddings, state_embeddings, action_embeddings], dim=1) # batch, block_size // 3, 3, n_embd
        token_embeddings = token_embeddings.permute(0, 2, 1, 3).reshape(-1, 3 * state_embeddings.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        
        # NOTE: durring inference, we need to predict next action. Here we still use same sequence length, because we can only get prediction from the state_embeddings
        x = self.ln_f(token_embeddings)
        #x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
            
        
        '''if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            # NOTE if cuda index error, please check if there is -1 in actions

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device) # size: (batch, block_size*3-1, n_embd), #NOTE: 因此真实的上下文长度上 Block Size * 3 
            # NOTE: 构造连续序列
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            if targets is None:
                token_embeddings[:,2::3,:] = action_embeddings[:,:-1,:] # here we drop last action
            else:
                token_embeddings[:,2::3,:] = action_embeddings
                
            context_timesteps = torch.repeat_interleave(timesteps, 3, dim=1)
            context_timesteps = context_timesteps[:, :token_embeddings.shape[1], :]

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # From 1, traj_length, n_embd to batch_size, traj_length, n_embd
        # TODO: change this to use the timestep from paper.
        
        gather_tensor =  torch.repeat_interleave(context_timesteps, self.config.n_embd, dim=-1)
        position_embeddings = torch.gather(all_global_pos_emb, 1, gather_tensor) + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss'''
        
    
    
    def get_action_prediction(self, states, actions, rtgs, timesteps):
        """
        Get action predictions from the GPT model given the input states, actions, returns-to-go (rtgs), and timesteps.

        Args:
            states (torch.Tensor): The state tensor of shape (batch_size, seq_length, state_dim).
            actions (torch.Tensor): The action tensor of shape (batch_size, seq_length, action_dim).
            rtgs (torch.Tensor): The returns-to-go tensor of shape (batch_size, seq_length, 1).
            timesteps (torch.Tensor): The timestep tensor of shape (batch_size, seq_length, 1).

        Returns:
            torch.Tensor: The predicted action for the last timestep of the first batch element,
                        of shape (action_dim,).
        """
        
        # reshape the input tensors to have a batch 1 
        states = states.reshape(1, -1, states.shape[-1])
        actions = actions.reshape(1, -1, actions.shape[-1])
        rtgs = rtgs.reshape(1, -1, rtgs.shape[-1])
        timesteps = timesteps.reshape(1, -1, 1)
        
        # Pad the input tensors if max_length is specified 
        if self.config.max_length is not None: 
            # Truncate the input tensors to the max_length
            states = states[:, -self.config.max_length:, :]
            actions = actions[:, -self.config.max_length:, :]
            rtgs = rtgs[:, -self.config.max_length:, :]
            timesteps = timesteps[:, -self.config.max_length:, :]
            
            # Here we create a attention mask to indicate the valid positions in the padded sequences 
            # Size of the padding tensors: (1, max_length - states.shape[1], states.shape[-1]), size of original tensors: (1, states.shape[1], states.shape[-1]), final size: (1, max_length, states.shape[-1])
            padding_states = torch.concatenate([torch.zeros((1, self.config.max_length - states.shape[1], states.shape[-1]), dtype=torch.float32, device=states.device), states], dim=1) 
            padding_actions = torch.concatenate([torch.zeros((1, self.config.max_length - actions.shape[1], actions.shape[-1]), dtype=torch.int64, device=actions.device), actions], dim=1)
            padding_rtgs = torch.concatenate([torch.zeros((1, self.config.max_length - rtgs.shape[1], rtgs.shape[-1]), dtype=torch.float32, device=rtgs.device), rtgs], dim=1)
            padding_timesteps = torch.concatenate([torch.zeros((1, self.config.max_length - timesteps.shape[1], timesteps.shape[-1]), dtype=torch.int64, device=timesteps.device), timesteps], dim=1)
            
            # for a padding_tensor, we have a mask value of it 
            attention_mask = torch.cat([torch.zeros((1, self.config.max_length - states.shape[1]), dtype=torch.int64, device=states.device), torch.ones((1, states.shape[1]), dtype=torch.int64, device=states.device)], dim=1).reshape(1, 1, 1, self.config.max_length)
            
        else: 
            attention_mask = None
            
        # now we can run the model in inference mode, no need to pass labels
        action_preds, _ = self.forward(padding_states, padding_actions, None, padding_rtgs, padding_timesteps)
        
        return action_preds
    def _top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out

    def sample_from_logits(self, logits, temperature=1.0, sample=False, top_k=None):
        """
        Given a sequence of logits, predict the next token in the sequence,
        feeding the predictions back into the model each time. This function
        assumes that the logits are already produced by the model and are
        passed directly to it.
        
        logti: (batch, block_size // 3, vocab_size)
        """
        # Assuming logits are of shape (b, v) where b is batch size, and v is vocabulary size
        # We only need the last logits for the next token prediction
        logits = logits[:, :] / temperature # (batch, vocab_size)
        
        # Optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = self._top_k_logits(logits, top_k)
        
        # Apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        
        # Return the index of the sampled token
        return ix
    
    def save(self, path): # TODO: add this to config 
        """
        Save the model's state_dict to a file.

        Args:
            path (str): The path where the model will be saved.
        """
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path, config):
        """
        Load the model's state_dict from a file and return an instance of the model.

        Args:
            path (str): The path where the model is saved.
            config: The configuration object for the model.

        Returns:
            GPT: An instance of the GPT model with the loaded parameters.
        """
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model