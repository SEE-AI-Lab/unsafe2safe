    const input = document.getElementById('photo');
    const uploadButton = document.getElementById('uploadButton');
    const cameraButton = document.getElementById('cameraButton');
    const captionStatus = document.getElementById('captionStatus');
    const privacyMarkers = document.getElementById('privacyMarkers');
    const editInstruction = document.getElementById('editInstruction');
    const instructionLoading = document.getElementById('instructionLoading');
    const preview = document.getElementById('preview');
    const cameraPreview = document.getElementById('cameraPreview');
    const photoActions = document.getElementById('photoActions');
    const cameraControls = document.getElementById('cameraControls');
    const cameraError = document.getElementById('cameraError');
    const edit = document.getElementById('edit');
    const resultImage = document.getElementById('resultImage');
    const resultStage = document.getElementById('resultStage');
    const placeholder = document.getElementById('placeholder');
    let selectedFile;
    let cameraStream;
    let selectedMode = 'spooky_private';
    let privacyReviewItems = [];
    const qrDialog = document.getElementById('qrDialog');
    const qrDialogTitle = document.getElementById('qrDialogTitle');
    const qrDialogImage = document.getElementById('qrDialogImage');
    const qrDialogLink = document.getElementById('qrDialogLink');
    const pipelineProgress = document.getElementById('pipelineProgress');
    const progressUpload = document.getElementById('progressUpload');
    const progressPhoto = document.getElementById('progressPhoto');
    const progressChoose = document.getElementById('progressChoose');
    const progressEdit = document.getElementById('progressEdit');
    const progressSteps = [progressUpload, progressPhoto, progressChoose, progressEdit];

    function setPipelineState(states) {
      let currentStep = states.findIndex(([state]) => state === 'running' || state === 'error');
      if (currentStep < 0) currentStep = states.findIndex(([state]) => state === 'idle');
      progressSteps.forEach((step, index) => {
        const [state, label] = states[index];
        step.dataset.state = state;
        step.classList.toggle('is-current', index === currentStep);
        step.classList.toggle('is-complete', state === 'complete');
        step.querySelector('.progress-state').textContent = label;
        step.toggleAttribute('aria-current', index === currentStep);
      });
      pipelineProgress.setAttribute('aria-label', currentStep < 0 ? 'Pipeline complete' : `Pipeline: ${progressSteps[currentStep].querySelector('strong').textContent}`);
    }

    function resetPipeline() {
      setPipelineState([
        ['idle', 'Waiting for photo'],
        ['idle', 'Waiting for upload'],
        ['idle', 'Waiting for public caption'],
        ['idle', 'Waiting for edit instruction']
      ]);
    }

    function renderCaption(element, text, changes, phraseKey, highlightClass) {
      const caption = text || '';
      const phrases = (changes || [])
        .map((change) => change[phraseKey])
        .filter(Boolean)
        .map((phrase) => ({ phrase, index: caption.indexOf(phrase) }))
        .filter((match) => match.index >= 0)
        .sort((a, b) => a.index - b.index);
      element.replaceChildren();
      let cursor = 0;
      phrases.forEach(({ phrase, index }) => {
        if (index < cursor) return;
        element.append(document.createTextNode(caption.slice(cursor, index)));
        const mark = document.createElement('mark');
        mark.className = `caption-highlight ${highlightClass}`;
        mark.textContent = phrase;
        element.append(mark);
        cursor = index + phrase.length;
      });
      element.append(document.createTextNode(caption.slice(cursor)));
    }

    function renderPrivacyMarkers(items) {
      privacyMarkers.replaceChildren();
      (items || []).forEach((item, index) => {
        const pin = document.createElement('button');
        pin.type = 'button';
        pin.className = 'privacy-pin';
        pin.style.left = `${Math.min(96, Math.max(4, Number(item.X) || 50))}%`;
        pin.style.top = `${Math.min(96, Math.max(4, Number(item.Y) || 50))}%`;
        pin.setAttribute('aria-expanded', 'false');
        pin.setAttribute('aria-label', `Privacy detail ${index + 1}: ${item.ITEM}. ${item.REASON}`);
        const tooltip = document.createElement('span');
        tooltip.className = 'privacy-tooltip';
        const title = document.createElement('strong');
        title.textContent = item.ITEM;
        const reason = document.createElement('span');
        reason.textContent = item.REASON;
        tooltip.append(title, reason);
        pin.append(tooltip);
        pin.addEventListener('click', () => {
          const open = pin.classList.toggle('is-open');
          pin.setAttribute('aria-expanded', String(open));
        });
        privacyMarkers.append(pin);
      });
    }

    function setImageRatio(width, height) {
      if (!width || !height) return;
      const ratio = `${width} / ${height}`;
      document.getElementById('drop').style.setProperty('--image-ratio', ratio);
      resultStage.style.setProperty('--image-ratio', ratio);
    }

    async function generateEditInstruction() {
      const publicCaption = document.getElementById('publicCaption').textContent;
      if (!selectedFile || !publicCaption) return;
      edit.disabled = true;
      editInstruction.hidden = true;
      instructionLoading.hidden = false;
      instructionLoading.dataset.state = 'running';
      instructionLoading.textContent = 'Gemini is generating a paper-guided instruction…';
      document.querySelectorAll('.choice').forEach((choice) => { choice.disabled = true; });
      setPipelineState([
        ['complete', 'Photo ready'],
        ['complete', 'Inspection & captions ready'],
        ['running', 'Generating edit instruction'],
        ['idle', 'Waiting for edit instruction']
      ]);
      const form = new FormData();
      form.append('public_caption', publicCaption);
      form.append('mode', selectedMode);
      form.append('privacy_review', JSON.stringify(privacyReviewItems));
      try {
        const response = await fetch('/edit-instruction', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'The edit instruction could not be generated.');
        editInstruction.textContent = data.edit_instruction;
        editInstruction.hidden = false;
        instructionLoading.hidden = true;
        instructionLoading.dataset.state = 'ready';
        edit.disabled = false;
        setPipelineState([
          ['complete', 'Photo ready'],
          ['complete', 'Inspection & captions ready'],
          ['complete', 'Edit instruction ready'],
          ['idle', 'Ready to generate safe image']
        ]);
      } catch (error) {
        instructionLoading.dataset.state = 'error';
        instructionLoading.textContent = `${error.message} Choose a mode again to retry.`;
        setPipelineState([
          ['complete', 'Photo ready'],
          ['complete', 'Inspection & captions ready'],
          ['error', 'Instruction generation failed'],
          ['idle', 'Waiting for edit instruction']
        ]);
      } finally {
        document.querySelectorAll('.choice').forEach((choice) => { choice.disabled = false; });
      }
    }

    document.querySelectorAll('[data-qr-id]').forEach((link) => link.addEventListener('click', (event) => {
      event.preventDefault();
      qrDialogTitle.textContent = link.dataset.qrTitle;
      qrDialogImage.src = `/static/qr/${link.dataset.qrId}.svg`;
      qrDialogImage.alt = `QR code for ${link.dataset.qrTitle}`;
      qrDialogLink.href = link.href;
      qrDialog.showModal();
    }));
    qrDialog.addEventListener('click', (event) => {
      if (!event.target.closest('.qr-dialog-layout > div, .qr-dialog-layout > img')) qrDialog.close();
    });

    uploadButton.addEventListener('click', () => input.click());
    cameraButton.addEventListener('click', startCamera);

    document.querySelectorAll('.choice').forEach((choice) => choice.addEventListener('click', () => {
      selectedMode = choice.dataset.mode;
      document.querySelectorAll('.choice').forEach((item) => item.setAttribute('aria-pressed', String(item === choice)));
      generateEditInstruction();
    }));

    const halloweenChoice = document.querySelector('.choice[data-mode="spooky_private"]');
    const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
    if (finePointer) {
      halloweenChoice.addEventListener('pointermove', (event) => {
        const bounds = halloweenChoice.getBoundingClientRect();
        const x = Math.max(50, Math.min(bounds.width - 50, event.clientX - bounds.left));
        const y = Math.max(52, Math.min(bounds.height - 52, event.clientY - bounds.top));
        halloweenChoice.style.setProperty('--bat-x', `${x}px`);
        halloweenChoice.style.setProperty('--bat-y', `${y}px`);
        halloweenChoice.classList.add('is-bat-flying');
      });
      halloweenChoice.addEventListener('pointerleave', () => {
        halloweenChoice.classList.remove('is-bat-flying');
      });
    }

    function stopCamera() {
      if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
      cameraPreview.srcObject = null;
      cameraPreview.hidden = true;
      document.getElementById('drop').classList.remove('camera-on');
      photoActions.hidden = false;
      cameraControls.hidden = true;
    }

    async function selectPhoto(file) {
      if (!file) return;
      stopCamera();
      if (preview.src.startsWith('blob:')) URL.revokeObjectURL(preview.src);
      selectedFile = file;
      privacyReviewItems = [];
      preview.src = URL.createObjectURL(file);
      preview.onload = () => setImageRatio(preview.naturalWidth, preview.naturalHeight);
      document.getElementById('drop').classList.add('has-image');
      privacyMarkers.replaceChildren();
      editInstruction.textContent = '';
      instructionLoading.hidden = false;
      instructionLoading.dataset.state = 'waiting';
      instructionLoading.textContent = 'Waiting for public caption';
      edit.disabled = true;
      resultImage.style.display = 'none';
      placeholder.style.display = 'block';
      placeholder.innerHTML = '<strong>Your safe image appears here</strong>Your generated image will appear here.';
      resultStage.classList.remove('processing');
      document.getElementById('download').style.display = 'none';
      document.getElementById('privateCaptionSkeleton').hidden = false;
      document.getElementById('publicCaptionSkeleton').hidden = false;
      document.getElementById('fullCaption').hidden = true;
      document.getElementById('publicCaption').hidden = true;
      document.getElementById('retryCaptions').hidden = true;
      await generateCaptions(file);
    }

    async function generateCaptions(file) {
      captionStatus.hidden = false;
      captionStatus.className = 'caption-status';
      captionStatus.textContent = 'Generating captions and privacy review…';
      document.getElementById('privateCaptionSkeleton').hidden = false;
      document.getElementById('publicCaptionSkeleton').hidden = false;
      document.getElementById('fullCaption').hidden = true;
      document.getElementById('publicCaption').hidden = true;
      document.getElementById('retryCaptions').hidden = true;
      setPipelineState([
        ['complete', 'Photo ready'],
        ['running', 'Reviewing privacy & generating captions'],
        ['idle', 'Waiting for public caption'],
        ['idle', 'Waiting for edit instruction']
      ]);
      uploadButton.disabled = true;
      cameraButton.disabled = true;
      const form = new FormData();
      form.append('photo', file);
      try {
        const response = await fetch('/caption', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'The captions could not be generated.');
        renderCaption(document.getElementById('fullCaption'), data.full_caption, data.privacy_changes, 'PRIVATE_PHRASE', 'is-private');
        renderCaption(document.getElementById('publicCaption'), data.public_caption, data.privacy_changes, 'PUBLIC_PHRASE', 'is-public');
        if (!data.public_caption) throw new Error('Stage 1 did not return a public caption. Try another photo.');
        privacyReviewItems = data.privacy_review || [];
        renderPrivacyMarkers(privacyReviewItems);
        document.getElementById('privateCaptionSkeleton').hidden = true;
        document.getElementById('publicCaptionSkeleton').hidden = true;
        document.getElementById('fullCaption').hidden = false;
        document.getElementById('publicCaption').hidden = false;
        captionStatus.hidden = true;
        setPipelineState([
          ['complete', 'Photo ready'],
          ['complete', 'Inspection & captions ready'],
          ['idle', 'Generating edit instruction'],
          ['idle', 'Waiting for edit instruction']
        ]);
        generateEditInstruction();
      } catch (error) {
        captionStatus.textContent = error.message;
        captionStatus.classList.add('error');
        document.getElementById('privateCaptionSkeleton').hidden = true;
        document.getElementById('publicCaptionSkeleton').hidden = true;
        document.getElementById('fullCaption').hidden = false;
        document.getElementById('publicCaption').hidden = false;
        document.getElementById('fullCaption').textContent = 'Captions could not be generated. Retry to try again.';
        document.getElementById('publicCaption').textContent = 'The public caption will appear after a successful retry.';
        document.getElementById('retryCaptions').hidden = false;
        setPipelineState([
          ['complete', 'Photo ready'],
          ['error', 'Captioning failed'],
          ['idle', 'Waiting for public caption'],
          ['idle', 'Waiting for edit instruction']
        ]);
      } finally {
        uploadButton.disabled = false;
        cameraButton.disabled = false;
      }
    }

    async function startCamera() {
      cameraButton.disabled = true;
      cameraError.hidden = true;
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' } },
          audio: false
        });
        cameraPreview.srcObject = cameraStream;
        cameraPreview.hidden = false;
        document.getElementById('drop').classList.add('camera-on');
        photoActions.hidden = true;
        cameraControls.hidden = false;
        await cameraPreview.play();
        setImageRatio(cameraPreview.videoWidth, cameraPreview.videoHeight);
      } catch {
        stopCamera();
        cameraError.hidden = false;
      } finally {
        cameraButton.disabled = false;
      }
    }

    input.addEventListener('change', () => {
      selectPhoto(input.files[0]);
      input.value = '';
    });

    document.getElementById('retryCaptions').addEventListener('click', () => generateCaptions(selectedFile));

    document.getElementById('captureButton').addEventListener('click', () => {
      const canvas = document.createElement('canvas');
      canvas.width = cameraPreview.videoWidth;
      canvas.height = cameraPreview.videoHeight;
      canvas.getContext('2d').drawImage(cameraPreview, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) selectPhoto(new File([blob], 'safeframe-camera.jpg', { type: 'image/jpeg' }));
      }, 'image/jpeg', 0.92);
    });

    document.getElementById('cancelCameraButton').addEventListener('click', stopCamera);

    edit.addEventListener('click', async () => {
      if (!selectedFile) return;
      edit.disabled = true;
      document.getElementById('editLabel').textContent = 'Generating safe image…';
      setPipelineState([
        ['complete', 'Photo ready'],
        ['complete', 'Inspection & captions ready'],
        ['complete', 'Edit instruction ready'],
        ['running', 'Generating safe image']
      ]);
      placeholder.innerHTML = '<strong>Generating safe image</strong>The public caption and edit instruction are guiding the edit.';
      placeholder.style.display = 'block';
      resultImage.style.display = 'none';
      resultStage.classList.add('processing');
      const form = new FormData();
      form.append('photo', selectedFile);
      form.append('public_caption', document.getElementById('publicCaption').textContent);
      form.append('edit_instruction', editInstruction.textContent);
      try {
        const response = await fetch('/edit', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'The edit could not be completed.');
        const url = `data:${data.mime_type};base64,${data.image}`;
        resultImage.src = url;
        resultImage.style.display = 'block';
        placeholder.style.display = 'none';
        resultStage.classList.remove('processing');
        const download = document.getElementById('download');
        download.href = url;
        download.style.display = 'inline-flex';
        setPipelineState([
          ['complete', 'Photo ready'],
          ['complete', 'Inspection & captions ready'],
          ['complete', 'Edit instruction ready'],
          ['complete', 'Safe image ready']
        ]);
      } catch (error) {
        placeholder.replaceChildren();
        const title = document.createElement('strong');
        title.textContent = 'Edit failed';
        placeholder.append(title, document.createTextNode(error.message));
        resultStage.classList.remove('processing');
        setPipelineState([
          ['complete', 'Photo ready'],
          ['complete', 'Inspection & captions ready'],
          ['complete', 'Edit instruction ready'],
          ['error', 'Safe image generation failed']
        ]);
      } finally {
        edit.disabled = false;
        document.getElementById('editLabel').textContent = 'Create this edit';
      }
    });

    resetPipeline();
