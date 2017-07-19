
import qprompt





def local_threshold_prompt(args):
    r = args
    block_sz = qprompt.ask_int("Block size")
    sigma = qprompt.ask_int("Sigma value")
    r.threshold_method = "local"
    r.block_sz = block_sz
    r.sigma = sigma

def set_threshold(r):
    menu = qprompt.Menu()
    menu.add("o", "Otsu thresholding")
    menu.add("i", "Isodata thresholding")
    menu.add("e", "Lee thresholding") 
    menu.add("l", "Local thresholding (slow)", local_threshold_prompt, [r])
    menu.add("q", "Quit")
    while menu.show(header = "Threshold Menu") != "q":
        pass
    #menu.main(loop=True)



if __name__ == "__main__":

    class Results(object):
        pass

    r = Results()
    
    set_threshold(r)

    if hasattr(r, "block_sz"):
        print r.block_sz
