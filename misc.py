def FEDMD_digest_revisit(dataloader, net, optimizer, device, config, aggregated_logits=None, mode="revisit"):
    epochs = config.get("epochs", 1)
    server_rounds = config.get("server_round", 0)
    proximal_mu = config.get("proximal_mu", 0.0)
    alpha = config.get("client_KD_alpha", 0.1)
    temperature = config.get("client_KD_temperature", 1.0)
    global_params = [params.detach().clone() for params in net.parameters()]
    Epoch_Loss = []
    Epoch_Accuracy = []

    def kd_loss(student_logits, labels, teacher_logits):
        loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits / temperature, dim=1), F.softmax(teacher_logits / temperature, dim=1)) * (temperature ** 2) + \
                                                F.cross_entropy(student_logits, labels) * (1 - alpha)
        return loss
    
    net.train()    
    
    if mode == "digest" and aggregated_logits is not None:
        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                student_logits = net(images)
                loss = kd_loss(student_logits, labels, aggregated_logits)
                loss.backward()
                optimizer.step()

    elif mode == "revisit":
        criterion = nn.CrossEntropyLoss()
        scheduler_ = config.get("scheduler", None)
        if scheduler_:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels) + proximal_loss_func(proximal_mu, net, global_params)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(dataloader.dataset)
            epoch_acc = 100. * correct / total
            if scheduler_ and epoch!=epochs-1:
                scheduler.step()
            
            Epoch_Loss.append(epoch_loss)
            Epoch_Accuracy.append(epoch_acc)

            print(f"Epoch {epoch + 1}/{epochs} \t Loss: {epoch_loss:.4f} \t Accuracy: {epoch_acc:.2f}%")



            

        
    