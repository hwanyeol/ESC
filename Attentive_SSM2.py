import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import matplotlib.pyplot as plt
import random

# ==========================================
# 0. 시드 고정
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
@dataclass
class Config:
    train_path: str = 'train1.csv'
    test_path: str = 'test1.csv'
    
    seq_len: int = 30           
    
    embedding_dim: int = 16    
    lstm_hidden_dim: int = 256
    lstm_layers: int = 2       
    attention_heads: int = 4   
    
    # [Trend 제거] Level(1) + Seasonality(7) = 8
    latent_dim: int = 8        
    
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 5
    
    device: str = (
        "mps" if torch.backends.mps.is_available() else 
        "cuda" if torch.cuda.is_available() else 
        "cpu"
    )

cfg = Config()
print(f"🚀 Running on device: {cfg.device}")

# ==========================================
# 2. 데이터셋 클래스
# ==========================================
class RossmannDataset(Dataset):
    def __init__(self, df, seq_len=30, is_train=True):
        self.seq_len = seq_len
        self.is_train = is_train
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(['Store', 'Date'], inplace=True)
        
        df['Sales_Log'] = np.log1p(df['Sales'])
        
        df['StateHoliday'] = df['StateHoliday'].astype(str).map({'0':0, 'a':1, 'b':2, 'c':3, '0.0':0}).fillna(0).astype(int)
        
        self.dynamic_cols = ['Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek']
        
        self.num_stores = df['Store'].max() + 1
        
        self.data_groups = {}
        for store_id, group in df.groupby('Store'):
            self.data_groups[store_id] = {
                'target': group['Sales_Log'].values.astype(np.float32),
                'static': store_id,
                'dynamic': group[self.dynamic_cols].values.astype(np.float32),
                'date': group['Date'].values
            }
            
        self.indices = []
        if is_train:
            for s_id, data in self.data_groups.items():
                t_len = len(data['target'])
                if t_len < seq_len: continue
                # Stride 14
                for t in range(0, t_len - seq_len + 1, 14):
                    self.indices.append((s_id, t))
            print(f"✅ Train sequences created: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        store_id, start_t = self.indices[idx]
        data = self.data_groups[store_id]
        end_t = start_t + self.seq_len
        
        z = data['target'][start_t:end_t]
        dyn = data['dynamic'][start_t:end_t]
        
        x_dow = dyn[:, 4].astype(np.int64)
        x_hol = dyn[:, 2].astype(np.int64)
        x_num = dyn[:, [0, 1, 3]] 
        
        return {
            'z': torch.tensor(z),
            'x_stat': torch.tensor(data['static'], dtype=torch.long),
            'x_dow': torch.tensor(x_dow),
            'x_hol': torch.tensor(x_hol),
            'x_num': torch.tensor(x_num)
        }

# ==========================================
# 3. Deep State Space Model (with Attention)
# ==========================================
class DeepStateModel(nn.Module):
    def __init__(self, cfg: Config, num_stores):
        super().__init__()
        self.cfg = cfg
        
        # Embeddings
        self.emb_store = nn.Embedding(num_stores, cfg.embedding_dim)
        self.emb_dow = nn.Embedding(8, 4)
        self.emb_hol = nn.Embedding(4, 4)
        
        rnn_input_dim = 3 + 4 + 4 + cfg.embedding_dim
        
        # LSTM
        self.lstm = nn.LSTM(rnn_input_dim, cfg.lstm_hidden_dim, cfg.lstm_layers, batch_first=True)
        
        # Attention Mechanism (Self-Attention)
        self.attention = nn.MultiheadAttention(embed_dim=cfg.lstm_hidden_dim, num_heads=cfg.attention_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.lstm_hidden_dim) 
        
        # Parameter Heads
        self.head_sigma_obs = nn.Sequential(nn.Linear(cfg.lstm_hidden_dim, 1), nn.Softplus())
        self.head_sigma_level = nn.Sequential(nn.Linear(cfg.lstm_hidden_dim, 1), nn.Softplus())
        # [Trend Head Removed]
        self.head_sigma_season = nn.Sequential(nn.Linear(cfg.lstm_hidden_dim, 1), nn.Softplus())
        
        self.head_mu_0 = nn.Linear(cfg.embedding_dim, cfg.latent_dim)
        self.head_sigma_0 = nn.Sequential(nn.Linear(cfg.embedding_dim, cfg.latent_dim), nn.Softplus())

    def forward(self, x_stat, x_dow, x_hol, x_num):
        B, T = x_dow.shape
        e_s = self.emb_store(x_stat).unsqueeze(1).expand(-1, T, -1)
        e_d = self.emb_dow(x_dow)
        e_h = self.emb_hol(x_hol)
        
        # Input Concat
        lstm_in = torch.cat([x_num, e_d, e_h, e_s], dim=2)
        
        # LSTM Pass
        lstm_out, _ = self.lstm(lstm_in)
        
        # Attention Pass (Query=Key=Value=lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual + Norm
        combined_out = self.norm(lstm_out + attn_out)
        
        # Heads
        mu_0 = self.head_mu_0(self.emb_store(x_stat))
        sigma_0 = self.head_sigma_0(self.emb_store(x_stat)) + 1e-4
        
        s_obs = self.head_sigma_obs(combined_out) + 1e-4
        s_lvl = self.head_sigma_level(combined_out) + 1e-4
        s_sea = self.head_sigma_season(combined_out) + 1e-4
        
        return mu_0, sigma_0, s_obs, s_lvl, s_sea

# ==========================================
# 4. Kalman Filter Loss (No Trend)
# ==========================================
def kalman_filter_loss(z, x_dow, x_num, mu_0, sigma_0, s_obs, s_lvl, s_sea, device):
    B, T = z.shape
    L = 8 # Level(1) + Season(7)
    
    # F: Identity Matrix (Trend 제거됨)
    F = torch.eye(L, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    
    m = mu_0.unsqueeze(2)
    P = torch.diag_embed(sigma_0**2)
    
    log_lik = 0
    valid_steps = 0 
    
    for t in range(T):
        is_open = x_num[:, t, 0].view(B, 1, 1) 
        
        a_t = torch.zeros(B, 1, L, device=device)
        a_t[:, 0, 0] = 1.0 # Level
        
        # Seasonality Index (1~7)
        day_idx = (x_dow[:, t] - 1) + 1 
        a_t.scatter_(2, day_idx.unsqueeze(1).unsqueeze(1), 1.0)
        
        # Q Matrix construction
        q_vals = torch.cat([s_lvl[:,t], s_sea[:,t].repeat(1,7)], dim=1)**2
        Q = torch.diag_embed(q_vals)
        
        # Predict
        m_pred = torch.bmm(F, m)
        P_pred = torch.bmm(torch.bmm(F, P), F.transpose(1,2)) + Q
        
        # Observation
        R = s_obs[:, t].view(B, 1, 1)**2
        y_pred = torch.bmm(a_t, m_pred)
        obs = z[:, t].view(B, 1, 1)
        residual = obs - y_pred
        
        S = torch.bmm(torch.bmm(a_t, P_pred), a_t.transpose(1,2)) + R
        K = torch.bmm(P_pred, a_t.transpose(1,2)) / (S + 1e-8)
        
        # Update
        m_updated = m_pred + torch.bmm(K, residual)
        I = torch.eye(L, device=device).unsqueeze(0)
        P_updated = torch.bmm(I - torch.bmm(K, a_t), P_pred)
        
        # Handle Closed Days (If closed, maintain prediction, don't update with observation)
        m = is_open * m_updated + (1 - is_open) * m_pred
        P = is_open * P_updated + (1 - is_open) * P_pred
        
        # Loss Calculation
        S_val = S.view(B)
        res_val = residual.view(B)
        term1 = torch.log(S_val + 1e-8)
        term2 = (res_val**2) / (S_val + 1e-8)
        step_loss = -0.5 * (term1 + term2 + np.log(2 * np.pi))
        
        mask = is_open.view(B)
        log_lik += (step_loss * mask).sum()
        valid_steps += mask.sum()
        
    return -log_lik / (valid_steps + 1e-8)

# ==========================================
# 5. 메인 파이프라인
# ==========================================
def run_pipeline():
    if not os.path.exists(cfg.train_path) or not os.path.exists(cfg.test_path):
        print("❌ Data files not found.")
        return

    print("📂 Loading Data...")
    train_df = pd.read_csv(cfg.train_path, low_memory=False)
    test_df = pd.read_csv(cfg.test_path, low_memory=False)
    
    train_ds = RossmannDataset(train_df, seq_len=cfg.seq_len, is_train=True)
    test_ds = RossmannDataset(test_df, is_train=False) 
    
    model = DeepStateModel(cfg, num_stores=train_ds.num_stores).to(cfg.device)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    print("\n🔥 Start Training...")
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    model.train()
    for epoch in range(cfg.epochs):
        ep_loss = 0
        steps = 0
        for b in loader:
            z = b['z'].to(cfg.device)
            xs = b['x_stat'].to(cfg.device)
            xd = b['x_dow'].to(cfg.device)
            xh = b['x_hol'].to(cfg.device)
            xn = b['x_num'].to(cfg.device)
            
            opt.zero_grad()
            mu0, sig0, so, sl, ss = model(xs, xd, xh, xn)
            
            loss = kalman_filter_loss(z, xd, xn, mu0, sig0, so, sl, ss, cfg.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            ep_loss += loss.item()
            steps += 1
        print(f"Epoch [{epoch+1}/{cfg.epochs}] Loss: {ep_loss/steps:.4f}")
        
    print("✨ Training Finished.")
    
    print("\n🔮 Starting Prediction...")
    model.eval()
    
    total_rmse = 0
    count = 0
    sample_plots = []
    
    # [수정] 전체 상점 대상
    test_stores = test_df['Store'].unique()
    total_stores = len(test_stores)
    
    print(f"총 {total_stores}개 상점에 대한 예측을 시작합니다.")

    with torch.no_grad():
        for i, store_id in enumerate(test_stores):
            # [요청하신 로그 포맷]
            print(f"Processing store {store_id} ({i+1}/{total_stores})...")
            
            if store_id not in train_ds.data_groups: continue
            
            train_data = train_ds.data_groups[store_id]
            test_data = test_ds.data_groups[store_id]
            
            ctx_len = 60
            z_ctx = torch.tensor(train_data['target'][-ctx_len:]).to(cfg.device)
            dyn_ctx = train_data['dynamic'][-ctx_len:]
            dyn_fut = test_data['dynamic']
            z_true = test_data['target'] 
            
            dyn_all = np.concatenate([dyn_ctx, dyn_fut], axis=0)
            
            x_dow = torch.tensor(dyn_all[:, 4].astype(int)).unsqueeze(0).to(cfg.device)
            x_hol = torch.tensor(dyn_all[:, 2].astype(int)).unsqueeze(0).to(cfg.device)
            x_num = torch.tensor(dyn_all[:, [0,1,3]]).unsqueeze(0).to(cfg.device)
            x_stat = torch.tensor([store_id]).to(cfg.device)
            
            mu0, sig0, so, sl, ss = model(x_stat, x_dow, x_hol, x_num)
            
            # C. 필터링 (Context)
            L = 8
            m = mu0.squeeze(0); P = torch.diag(sig0.squeeze(0)**2)
            F = torch.eye(L, device=cfg.device)
            
            T_ctx = len(z_ctx)
            for t in range(T_ctx):
                is_open = x_num[0, t, 0]
                
                a_t = torch.zeros(L, device=cfg.device); a_t[0]=1.0
                day_idx = (x_dow[0,t]-1)+1
                a_t[day_idx] = 1.0
                
                q_d = torch.cat([sl[0,t], ss[0,t].repeat(7)])**2
                Q = torch.diag(q_d)
                
                m_pred = F @ m
                P_pred = F @ P @ F.T + Q
                
                if is_open == 1:
                    R = so[0,t]**2
                    y_p = torch.dot(a_t, m_pred)
                    K = (P_pred @ a_t) / (torch.dot(a_t, P_pred @ a_t) + R + 1e-8)
                    m = m_pred + K * (z_ctx[t] - y_p)
                    P = (torch.eye(L, device=cfg.device) - torch.outer(K, a_t)) @ P_pred
                else:
                    m, P = m_pred, P_pred
                
            # D. 예측 (Future Sampling)
            T_fut = len(dyn_fut)
            NUM_SAMPLES = 32
            samples = np.zeros((NUM_SAMPLES, T_fut))
            
            m_rt, P_rt = m, P 
            
            for k in range(NUM_SAMPLES):
                l_t = m_rt + torch.randn_like(m_rt) * torch.sqrt(torch.diagonal(P_rt))
                for t in range(T_fut):
                    t_idx = T_ctx + t
                    is_open = x_num[0, t_idx, 0]
                    
                    if is_open == 0:
                        samples[k, t] = 0.0
                        q_d = torch.cat([sl[0,t_idx], ss[0,t_idx].repeat(7)])
                        l_t += q_d * torch.randn_like(l_t) 
                        continue

                    a_t = torch.zeros(L, device=cfg.device); a_t[0]=1.0
                    day_idx = (x_dow[0,t_idx]-1)+1
                    a_t[day_idx] = 1.0
                    
                    q_d = torch.cat([sl[0,t_idx], ss[0,t_idx].repeat(7)])
                    l_t += q_d * torch.randn_like(l_t)
                    
                    y_mean = torch.dot(a_t, l_t)
                    sigma_val = torch.clamp(so[0, t_idx], max=1.0)
                    dist_val = y_mean + sigma_val * torch.randn(1).to(cfg.device)
                    samples[k, t] = dist_val.item()
            
            pred_mean_log = np.median(samples, axis=0)
            pred_mean = np.expm1(pred_mean_log)
            actual = np.expm1(z_true)
            
            mse = np.mean((pred_mean - actual)**2)
            total_rmse += np.sqrt(mse)
            count += 1
            
            if len(sample_plots) < 5:
                sample_plots.append({
                    'id': store_id, 'true': actual, 'pred': pred_mean, 
                    'date': test_data['date']
                })
                
    print(f"\n📊 Evaluation Result")
    print(f"Average RMSE over {count} stores: {total_rmse/count:.4f}")
    
    target_plots = sample_plots[:50] 

    if target_plots:
        # 1. 이미지를 저장할 폴더 생성 (없으면 만듦)
        save_dir = 'forecast_results'
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 Saving first 50 plots to '{save_dir}/' folder...")

        for i, p in enumerate(target_plots):
            # 2. 개별 그래프 그리기
            plt.figure(figsize=(12, 4)) 
            plt.plot(p['date'], p['true'], label='Actual', color='black')
            plt.plot(p['date'], p['pred'], label='Forecast', color='blue', linestyle='--')
            plt.title(f"Store {p['id']} Forecast")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 3. 개별 파일로 저장 (예: forecast_results/store_1.png)
            plt.savefig(f"{save_dir}/store_{p['id']}.png")
            
            # 4. 중요: 메모리 해제 (안 하면 메모리 터짐)
            plt.close() 
if __name__ == '__main__':
    run_pipeline()