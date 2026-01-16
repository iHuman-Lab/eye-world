# -------------------------------------------------
# Lightning V-JEPA 2 World Model
# -------------------------------------------------
class LightningVJEPA(pl.LightningModule):
    def __init__(
        self,
        student_patch_dim,
        teacher_patch_dim,
        config,
        embed_dim=768,
        depth=12,
        predictor_depth=4,
        heads=12,
        mlp_dim=3072,
        mask_ratio=0.6,
        lr=1e-4,
        ema_decay=0.996,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay

        # -----------------------------
        # Patch embeddings
        # -----------------------------
        self.student_patch_embed = VideoPatchEmbedding(student_patch_dim, embed_dim)
        self.teacher_patch_embed = VideoPatchEmbedding(teacher_patch_dim, embed_dim)

        # -----------------------------
        # Transformers
        # -----------------------------
        self.student = TransformerEncoder(embed_dim, depth, heads, mlp_dim)
        self.teacher = TransformerEncoder(embed_dim, depth, heads, mlp_dim)
        self.predictor = TransformerEncoder(embed_dim, predictor_depth, heads, mlp_dim)

        # initialize teacher
        self._init_teacher()
        for p in self.teacher.parameters():
            p.requires_grad = False

        '''
        # -------------------------------------------------
        def forward(self, stacked_frames):
            """
            stacked_frames: [T, C, H, W]  -> assume already grayscale and stacked
            Splits context vs target frames
            """
            T = stacked_frames.shape[0]
            context_frames = stacked_frames[:-1]  # student sees all except last
            target_frame = stacked_frames[-1:]  # teacher sees last frame

            if DEBUG:
                print("[forward] stacked shape:", stacked_frames.shape)
                print("  context_frames shape:", context_frames.shape)
                print("  target_frame shape:", target_frame.shape)

            # Patch embedding
            context_tokens = stacked_frames[:3]  # frames t-3, t-2, t-1
            target_tokens = stacked_frames[:4]  # frames t-3, t-2, t-1, t
            return context_tokens, target_tokens

            '''

    def forward(self, stacked_frames):
        """
        stacked_frames: [T, C, H, W]  -> assume already grayscale and stacked
        Splits context vs target frames for student/teacher
        """

        print("\n[forward] Received stacked_frames")
        print("  stacked_frames shape:", stacked_frames.shape)

        T = stacked_frames.shape[0]

        if T < 2:
            print("  WARNING: less than 2 frames received!")

        # Split frames
        context_frames = stacked_frames[:-1]  # student sees all except last
        target_frame = stacked_frames[-1:]  # teacher sees last frame

        print("  context_frames shape:", context_frames.shape)
        print("  target_frame shape:", target_frame.shape)
        print(
            "  context_frames content (min/max):",
            context_frames.min().item(),
            context_frames.max().item(),
        )
        print(
            "  target_frame content (min/max):",
            target_frame.min().item(),
            target_frame.max().item(),
        )

        # Patch embedding
        # student sees frames t-3, t-2, t-1
        # teacher sees frames t-3, t-2, t-1, t
        context_tokens = stacked_frames[:3]
        target_tokens = stacked_frames[:4]

        print("  context_tokens shape:", context_tokens.shape)
        print("  target_tokens shape:", target_tokens.shape)

        return context_tokens, target_tokens

        # --------------------------
        # Patch embedding

    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        stacked, gaze = batch  # stacked: [4, C, H, W]

        # -----------------------------
        # Forward: patch embedding
        # -----------------------------
        student_tokens, teacher_tokens = self.forward(stacked)
        # shapes: [N, 768]

        student_tokens = student_tokens.unsqueeze(0)  # [1, N, 768]
        teacher_tokens = teacher_tokens.unsqueeze(0)  # [1, N, 768]

        # -----------------------------
        # Mask patches (token-level)
        # -----------------------------
        N = student_tokens.size(1)
        mask = random_patch_mask(N, self.mask_ratio, student_tokens.device)

        context_tokens = student_tokens[:, ~mask]  # [1, Nc, 768]
        target_tokens = teacher_tokens[:, mask]  # [1, Nt, 768]

        # -----------------------------
        # Student + predictor
        # -----------------------------
        student_repr = self.student(context_tokens)
        pred = self.predictor(student_repr)

        # -----------------------------
        # Teacher (EMA, no grad)
        # -----------------------------
        with torch.no_grad():
            teacher_repr = self.teacher(target_tokens)

        # -----------------------------
        #    Loss
        # -----------------------------
        loss = F.mse_loss(
            F.normalize(pred, dim=-1),
            F.normalize(teacher_repr, dim=-1),
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------------------------------------------------
    def _init_teacher(self):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data.copy_(s.data)

    def update_teacher(self):
        """EMA update of teacher weights"""
        with torch.no_grad():
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                t.mul_(self.ema_decay).add_(s, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    # -------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.student_patch_embed.parameters())
            + list(self.student.parameters())
            + list(self.predictor.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
