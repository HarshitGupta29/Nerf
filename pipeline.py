import nerf_model
import extras
import torch
import load_data

def run(epoch=1000, batch_size=64):
    #TODO: Load data here using load_data
    model = nerf_model.nerf()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=e-10)
    for i in range(epoch):
        print("Running epoch {}".format(i))
        ################################################################################################
        ## prepare for one iteration of nerf ##
        o, d = extras.ray(height, width, focal_length, pose)
        x = #TODO
        gamma = extras.positional_encoding(x)
        ground_truth = #TODO
        ################################################################################################
        pred = model.forward(gamma)
        loss  = torch.nn.functional.mse_loss(pred, ground_truth)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    run()
