## NOTE: This task is strongly required to progress, so a solution is given

from transformers import BlipProcessor, BlipForConditionalGeneration

## TODO: Import the model components
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

text_img_pairs = []

for f in sorted(glob("img-files/*")):
    print(f, blip_pipe(f), sep='\n')

    raw_image = PIL.Image.open(f)

    ## TODO: Feel free to change the string. We found it good for first run
    text = "I love cats " 
    # TODO: Perform conditional image captioning and print response
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    response = processor.decode(out[0], skip_special_tokens=True)
    print(response, '\n')
    text_img_pairs += [(response, raw_image)]
    
## NOTE: The next section depends on this, since text_img_pairs should be aggregated