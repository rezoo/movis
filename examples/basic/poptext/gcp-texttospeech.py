import json

from google.cloud import texttospeech_v1beta1 as tts

# NOTE!
# This example uses the Google Cloud Text-to-Speech API. You will need
# authentication already configured and this may also incur charges
# against your GCP billing account.


def main():
    client = tts.TextToSpeechClient()

    text_input = """
    <speak>
        I'm using python<mark name="I'm using python"/>
        to generate<mark name="to generate"/>
        pop text!<mark name="pop text"/>
    </speak>
    """

    voice = tts.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-A",
        ssml_gender=tts.SsmlVoiceGender.NEUTRAL,
    )

    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.OGG_OPUS,
        sample_rate_hertz=48000,
    )

    synthesis_request = tts.SynthesizeSpeechRequest(
        input=tts.SynthesisInput(ssml=text_input),
        voice=voice,
        audio_config=audio_config,
        enable_time_pointing=[tts.SynthesizeSpeechRequest.TimepointType.SSML_MARK],
    )

    response = client.synthesize_speech(request=synthesis_request)

    with open("output.ogg", "wb") as f:
        f.write(response.audio_content)

    timepoints = []

    for timepoint in response.timepoints:
        timepoints.append(
            {
                "name": timepoint.mark_name,
                "time_seconds": timepoint.time_seconds,
            }
        )

    with open("timings.json", "w") as f:
        json.dump(timepoints, f)


if __name__ == '__main__':
    main()
