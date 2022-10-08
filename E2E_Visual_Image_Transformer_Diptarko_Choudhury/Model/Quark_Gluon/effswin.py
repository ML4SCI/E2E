import timm
import torch
import torch.nn as nn



#Stage 1> A clean Swin transformer model will be trained trained
#Stage 2> The swin Transformer blocks are frozen and the new embeding layer is attached and trained
#The old embedding blocks are replaced by the CNNs.In this stage only the embeding layer is trained
#Stage 3> The entire model is unfrozen and trained



class Hybrid_embed(nn.Module):
    def __init__(self, feature_model, img_size, channels, efn_blocks, dims):
        super().__init__()


        self.feature_extractor = timm.create_model(feature_model,
                                                   features_only=True,
                                                   out_indices=[efn_blocks])


        self.feature_extractor.conv_stem = nn.Conv2d(3,
                                       40,
                                       kernel_size=(3, 3),
                                       stride=(4, 4),
                                       padding=(1, 1),
                                       bias=False)

        with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = self.feature_extractor.training
                if training:
                    self.feature_extractor.eval()
                o = self.feature_extractor(torch.zeros(1, channels, img_size[0], img_size[1]))
                self.channel_output = o[0].shape[1]
                self.feature_extractor.train(training)

        self.embed_matcher = nn.Sequential(
            nn.Conv2d(self.channel_output, dims, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dims, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True)
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.embed_matcher(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Hybrid_swin_effnet(nn.Module):
    def __init__(self, feature_model = "efficientnet_b3",img_size = (224,224), channels = 3, efn_blocks = 2, swin_blocks = 2, no_classes = 1):
        super().__init__()
        assert efn_blocks + swin_blocks == 4,f"The total no of blocks must be 4, instead {efn_blocks+swin_blocks} blocks provided "
        self.s1_flag = True
        self.s2_flag = True
        self.s3_flag = True
        self.swin_blocks = swin_blocks
        self.swin_backbone = timm.create_model("swin_tiny_patch4_window7_224")
        self.embeded_dim = self.swin_backbone.embed_dim * (2**(4 - self.swin_blocks))
        self.Hybrid_patch_embed = Hybrid_embed(feature_model = "efficientnet_b3",
                                                      img_size = (224,224),
                                                      channels = 3,
                                                      efn_blocks = 2,
                                                      dims = self.embeded_dim)
        self.swin_backbone.head = nn.Linear(self.swin_backbone.num_features, no_classes)


    def forward(self, image, stage):
        """
        The forward method is a bit unuasual as it also accepts a stage arugument
        the stage argument is passed with numbers 2 or 3 when the stage is to be
        changed
        """

        if stage == 2:
            #Attaching the new embeding layer

            self.swin_backbone.patch_embed = self.Hybrid_patch_embed

            for i in range((4- self.swin_blocks)):
                self.swin_backbone.layers[i] = nn.Identity()

            # Freezing the swin layers

            for layer in self.swin_backbone.layers:
                for para in layer.parameters():
                    para.requires_grad = False

           #Freezing the head
            for para in self.swin_backbone.head.parameters():
                para.requires_grad = False

        if stage == 3:
            #Unfreezing the network
            for para in self.swin_backbone.parameters():
                para.requires_grad = True

        return self.swin_backbone(image).squeeze()
