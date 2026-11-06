import {
  AmbientLight, Color, DirectionalLight, Mesh, NoToneMapping, OrthographicCamera,
  PCFShadowMap, PlaneGeometry, Scene, ShaderMaterial, ShadowMaterial,
  Spherical, SRGBColorSpace, Vector2, Vector3, VSMShadowMap,
  WebGLRenderer, WebGLRenderTarget,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { clamp } from './performance.js';

const ELEVATION = 0.185;
const PIXELS_PER_UNIT = 242.75;
const STUDIO_COLOR = 0xcdcdcd;
const REFERENCE_HEIGHT = 1350 / PIXELS_PER_UNIT;
const TARGET_Y = 1.375;

function lightStudio(scene, softShadows) {
  scene.add(new AmbientLight(0xffffff, Math.PI * 0.325));
  const bounce = new DirectionalLight(0xffffff, 0.48);
  bounce.position.set(-0.08, 0.20, 1);
  scene.add(bounce);

  const key = new DirectionalLight(0xffffff, 2.13);
  key.target.position.set(0, 1.15, 0);
  key.position.set(-0.43, 1, 0.54).multiplyScalar(7).add(key.target.position);
  key.castShadow = true;
  key.shadow.mapSize.set(768, 768);
  Object.assign(key.shadow.camera, { left: -2.95, right: 3.25, top: 3.0, bottom: -3.0, near: 0.4, far: 15 });
  key.shadow.camera.updateProjectionMatrix();
  key.shadow.normalBias = 0.018;
  key.shadow.bias = -0.00022;
  key.shadow.radius = softShadows ? 10 : 3;
  key.shadow.blurSamples = 12;
  key.shadow.intensity = 0.95;
  scene.add(key, key.target);
}

function createFloor(texture) {
  const geometry = new PlaneGeometry(260, 260);
  const material = new ShaderMaterial({
    name: 'Satin studio floor',
    uniforms: {
      studioColor: { value: new Color(STUDIO_COLOR) },
      reflection: { value: texture },
      screenSize: { value: new Vector2(1, 1) },
      viewElevation: { value: Math.sin(ELEVATION) },
    },
    vertexShader: `
      varying vec2 floorPoint;
      void main() {
        floorPoint = (modelMatrix * vec4(position, 1.0)).xz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 studioColor;
      uniform sampler2D reflection;
      uniform vec2 screenSize;
      uniform float viewElevation;
      varying vec2 floorPoint;

      vec3 reflectedAt(vec2 uv) {
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return studioColor;
        return texture2D(reflection, uv).rgb;
      }

      float footDistance(float x) {
        return length((floorPoint - vec2(x, 0.125)) / vec2(0.232, 0.300));
      }

      void main() {
        float away = length(floorPoint);
        float satin = 0.48 * pow(1.0 - clamp(viewElevation, 0.0, 1.0), 1.25) *
                      (1.0 - smoothstep(1.0, 8.0, away));
        vec3 color = studioColor;
        if (satin > 0.002) {
          vec2 screen = gl_FragCoord.xy / screenSize;
          vec2 uv = vec2(1.0 - screen.x, screen.y);
          vec2 blur = vec2(0.043 * screenSize.y / screenSize.x, 0.034) *
                      mix(0.68, 1.16, smoothstep(0.4, 4.0, away));
          vec3 reflected = reflectedAt(uv) * 0.204;
          reflected += 0.124 * (reflectedAt(uv + vec2(blur.x, 0.0)) + reflectedAt(uv - vec2(blur.x, 0.0)) +
                                reflectedAt(uv + vec2(0.0, blur.y)) + reflectedAt(uv - vec2(0.0, blur.y)));
          reflected += 0.075 * (reflectedAt(uv + blur) + reflectedAt(uv - blur) +
                                reflectedAt(uv + vec2(blur.x, -blur.y)) + reflectedAt(uv + vec2(-blur.x, blur.y)));
          color = mix(studioColor, min(studioColor, reflected), satin);
        }
        float foot = min(footDistance(-0.243), footDistance(0.243));
        float contact = 0.22 * (1.0 - smoothstep(0.72, 1.66, foot)) +
                        0.36 * (1.0 - smoothstep(0.84, 1.20, foot));
        gl_FragColor = vec4(color * (1.0 - contact), 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
  const floor = new Mesh(geometry, material);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.006;
  const shadow = new Mesh(geometry, new ShadowMaterial({
    color: 0x222222, opacity: 0.41, depthWrite: false,
  }));
  shadow.rotation.x = floor.rotation.x;
  shadow.position.y = -0.004;
  shadow.receiveShadow = true;
  return { floor, shadow };
}

export class Studio {
  constructor(canvas, container) {
    this.container = container;
    this.scene = new Scene();
    this.scene.background = new Color(STUDIO_COLOR);
    this.renderer = new WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = NoToneMapping;
    const softShadows = this.renderer.extensions.has('EXT_color_buffer_float');
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = softShadows ? VSMShadowMap : PCFShadowMap;
    this.renderer.shadowMap.autoUpdate = false;
    lightStudio(this.scene, softShadows);

    this.camera = new OrthographicCamera(-2, 2, 2.8, -2.8, 0.1, 120);
    this.camera.position.set(0, TARGET_Y + Math.sin(ELEVATION)*9, Math.cos(ELEVATION)*9);
    const controls = new OrbitControls(this.camera, canvas);
    controls.target.set(0, TARGET_Y, 0);
    controls.enableDamping = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    controls.dampingFactor = 0.11;
    controls.enablePan = false;
    controls.rotateSpeed = 0.69;
    controls.zoomSpeed = 0.68;
    controls.minPolarAngle = 0.16;
    controls.maxPolarAngle = Math.PI / 2 - 0.052;
    controls.minZoom = 0.70;
    controls.maxZoom = 2.8;
    controls.update();
    controls.saveState();
    this.controls = controls;
    controls.addEventListener('start', () => { container.dataset.dragging = 'true'; });
    controls.addEventListener('end', () => { delete container.dataset.dragging; });
    canvas.addEventListener('keydown', event => this.onViewKey(event));

    this.reflectionTarget = new WebGLRenderTarget(1, 1);
    this.mirrorCamera = new OrthographicCamera();
    this.viewDirection = new Vector3();
    this.mirrorTarget = new Vector3();
    this.lastShadowAt = -Infinity;
    const { floor, shadow } = createFloor(this.reflectionTarget.texture);
    this.floor = floor;
    this.shadow = shadow;
    this.scene.add(floor, shadow);
    this.resize();
    this.resizer = new ResizeObserver(() => this.resize());
    this.resizer.observe(container);
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    const aspect = width / height;
    const halfHeight = REFERENCE_HEIGHT * 0.5;
    const halfWidth = halfHeight * aspect;
    const centerX = 19 / PIXELS_PER_UNIT * Math.min(1, aspect / 0.8);
    const centerY = (1070 - 675) / PIXELS_PER_UNIT - TARGET_Y * Math.cos(ELEVATION);
    Object.assign(this.camera, {
      left: -halfWidth + centerX, right: halfWidth + centerX,
      top: halfHeight + centerY, bottom: -halfHeight + centerY,
    });
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.6));
    this.renderer.setSize(width, height, false);
    const size = this.floor.material.uniforms.screenSize.value;
    this.renderer.getDrawingBufferSize(size);
    const scale = Math.min(0.54, 900 / Math.max(size.x, size.y));
    this.reflectionTarget.setSize(Math.max(1, Math.round(size.x*scale)), Math.max(1, Math.round(size.y*scale)));
  }

  reset() {
    this.controls.reset();
  }

  onViewKey(event) {
    const direction = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
    if (direction) {
      const offset = this.camera.position.clone().sub(this.controls.target);
      const sphere = new Spherical().setFromVector3(offset);
      sphere.theta += direction[0] * 0.075;
      sphere.phi = clamp(sphere.phi + direction[1] * 0.075,
        this.controls.minPolarAngle, this.controls.maxPolarAngle);
      this.camera.position.copy(offset.setFromSpherical(sphere)).add(this.controls.target);
      this.controls.update();
    } else if (['+', '=', '-'].includes(event.key)) {
      this.camera.zoom = clamp(this.camera.zoom * (event.key === '-' ? 1/1.12 : 1.12),
        this.controls.minZoom, this.controls.maxZoom);
      this.camera.updateProjectionMatrix();
    } else if (event.key === 'Home') {
      this.reset();
    } else {
      return;
    }
    event.preventDefault();
  }

  render(now) {
    this.controls.update();
    this.camera.updateMatrixWorld();
    this.camera.getWorldDirection(this.viewDirection);
    this.floor.material.uniforms.viewElevation.value = Math.max(0, -this.viewDirection.y);

    // A camera reflected in the floor also reverses its horizontal basis.
    // Flip the orthographic bounds here and sample the target mirrored in X.
    const mirror = this.mirrorCamera.copy(this.camera, false);
    mirror.position.y *= -1;
    mirror.up.set(0, 1, 0).applyQuaternion(this.camera.quaternion);
    mirror.up.y *= -1;
    this.viewDirection.y *= -1;
    mirror.lookAt(this.mirrorTarget.copy(mirror.position).add(this.viewDirection));
    mirror.left = -this.camera.right;
    mirror.right = -this.camera.left;
    mirror.updateProjectionMatrix();

    this.floor.visible = this.shadow.visible = false;
    this.renderer.setRenderTarget(this.reflectionTarget);
    if (now - this.lastShadowAt >= 30) {
      this.renderer.shadowMap.needsUpdate = true;
      this.lastShadowAt = now;
    }
    this.renderer.render(this.scene, mirror);
    this.renderer.setRenderTarget(null);
    this.floor.visible = this.shadow.visible = true;
    this.renderer.render(this.scene, this.camera);
  }
}
