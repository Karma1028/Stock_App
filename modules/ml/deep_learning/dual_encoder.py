"""
Dual-Encoder architecture combining price sequences and text embeddings.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
import pickle

class PriceEncoder(nn.Module):
    """
    LSTM encoder for price time-series data.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(PriceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        output, (hidden, cell) = self.lstm(x)
        # Return the last hidden state
        return hidden[-1]  # (batch, hidden_size)

class TextEncoder(nn.Module):
    """
    Dense encoder for text embeddings (from sentence-transformers).
    """
    def __init__(self, embedding_size=384, hidden_size=64):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Linear(embedding_size, 128)
        self.fc2 = nn.Linear(128, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, embedding_size)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # (batch, hidden_size)

class DualEncoderModel(nn.Module):
    """
    Combines price and text encoders with a fusion layer.
    """
    def __init__(self, price_input_size, text_embedding_size=384, hidden_size=64):
        super(DualEncoderModel, self).__init__()
        
        self.price_encoder = PriceEncoder(price_input_size, hidden_size)
        self.text_encoder = TextEncoder(text_embedding_size, hidden_size)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predict single value (5-day return)
        )
        
    def forward(self, price_seq, text_emb):
        # Encode price sequence
        price_features = self.price_encoder(price_seq)
        
        # Encode text
        text_features = self.text_encoder(text_emb)
        
        # Concatenate and fuse
        combined = torch.cat([price_features, text_features], dim=1)
        output = self.fusion(combined)
        
        return output

class StockDataset(Dataset):
    """
    PyTorch Dataset for dual-encoder training.
    """
    def __init__(self, price_sequences, text_embeddings, targets):
        self.price_sequences = torch.FloatTensor(price_sequences)
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.price_sequences[idx],
            self.text_embeddings[idx],
            self.targets[idx]
        )

class DualEncoderTrainer:
    """
    Training wrapper for the dual-encoder model.
    """
    def __init__(self):
        self.models_dir = Config.MODELS_DIR
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_sequences(self, df_features, seq_length=30):
        """
        Prepares price sequences and text embeddings for training.
        """
        # Select price features
        price_cols = [
            'Returns_1d', 'Vol_21d', 'MA_21d', 'RSI', 'MACD', 'sentiment_score'
        ]
        price_cols = [c for c in price_cols if c in df_features.columns]
        
        # Get text embeddings if available
        emb_cols = [c for c in df_features.columns if c.startswith('emb_')]
        
        price_sequences = []
        text_embeddings = []
        targets = []
        
        for i in range(seq_length, len(df_features)):
            # Price sequence
            seq = df_features[price_cols].iloc[i-seq_length:i].values
            price_sequences.append(seq)
            
            # Text embedding (use latest available)
            if emb_cols:
                emb = df_features[emb_cols].iloc[i].values
            else:
                emb = np.zeros(384)  # Default embedding size
            text_embeddings.append(emb)
            
            # Target (5-day return)
            if 'Target_5d' in df_features.columns:
                target = df_features['Target_5d'].iloc[i]
                targets.append(target)
        
        return np.array(price_sequences), np.array(text_embeddings), np.array(targets)
    
    def train(self, df_features, epochs=20, batch_size=32, lr=0.001):
        """
        Trains the dual-encoder model.
        """
        print("🧠 Training Dual-Encoder Model...")
        
        # Prepare data
        price_seq, text_emb, targets = self.prepare_sequences(df_features)
        
        # Remove NaN targets
        valid_idx = ~np.isnan(targets)
        price_seq = price_seq[valid_idx]
        text_emb = text_emb[valid_idx]
        targets = targets[valid_idx]
        
        if len(targets) < 50:
            print("Not enough data for training")
            return None
        
        # Train/val split
        split_idx = int(len(targets) * 0.8)
        train_dataset = StockDataset(
            price_seq[:split_idx],
            text_emb[:split_idx],
            targets[:split_idx]
        )
        val_dataset = StockDataset(
            price_seq[split_idx:],
            text_emb[split_idx:],
            targets[split_idx:]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        price_input_size = price_seq.shape[2]
        text_emb_size = text_emb.shape[1]
        
        self.model = DualEncoderModel(price_input_size, text_emb_size).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for price_batch, text_batch, target_batch in train_loader:
                price_batch = price_batch.to(self.device)
                text_batch = text_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(price_batch, text_batch).squeeze()
                loss = criterion(predictions, target_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for price_batch, text_batch, target_batch in val_loader:
                    price_batch = price_batch.to(self.device)
                    text_batch = text_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                    
                    predictions = self.model(price_batch, text_batch).squeeze()
                    loss = criterion(predictions, target_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print("✅ Training complete!")
        return self.model
    
    def predict(self, df_features):
        """
        Generates predictions using the trained model.
        """
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return None
        
        self.model.eval()
        
        price_seq, text_emb, _ = self.prepare_sequences(df_features)
        
        with torch.no_grad():
            price_tensor = torch.FloatTensor(price_seq).to(self.device)
            text_tensor = torch.FloatTensor(text_emb).to(self.device)
            
            predictions = self.model(price_tensor, text_tensor).cpu().numpy().squeeze()
        
        return predictions
    
    def save_model(self):
        """Saves the model to disk."""
        model_path = self.models_dir / "dual_encoder.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        """Loads the model from disk."""
        model_path = self.models_dir / "dual_encoder.pt"
        if not model_path.exists():
            print("No saved model found")
            return None
        
        # Need to know architecture to load
        # This is a simplified version - in production, save architecture params too
        print(f"Loading model from {model_path}")
        # Implementation would require saving model config
        return None
