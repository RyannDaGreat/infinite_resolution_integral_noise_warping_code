/**
 * WebGL 2 rendering: context, FBOs, shader programs, draw calls.
 * Port of game_demo/renderer.py.
 */

import { cubeVertices, floorVertices, quadVertices } from './geometry.js';
import { sceneVert, sceneFrag, displayVert, displayFrag } from './shaders.js';

const { mat4 } = window.glMatrix;

/**
 * Compile a shader from source. Throws on error.
 * @param {WebGL2RenderingContext} gl
 * @param {number} type - gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
 * @param {string} source
 * @returns {WebGLShader}
 */
function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error('Shader compile error: ' + info);
    }
    return shader;
}

/**
 * Link a program from vertex + fragment shaders. Throws on error.
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLShader} vert
 * @param {WebGLShader} frag
 * @returns {WebGLProgram}
 */
function linkProgram(gl, vert, frag) {
    const prog = gl.createProgram();
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(prog);
        gl.deleteProgram(prog);
        throw new Error('Program link error: ' + info);
    }
    return prog;
}

/**
 * Create a shader program from vertex + fragment source strings.
 * @param {WebGL2RenderingContext} gl
 * @param {string} vertSrc
 * @param {string} fragSrc
 * @returns {WebGLProgram}
 */
function createProgram(gl, vertSrc, fragSrc) {
    const v = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
    const f = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
    return linkProgram(gl, v, f);
}

export class Renderer {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {number} width
     * @param {number} height
     */
    constructor(canvas, width, height) {
        this.width = width;
        this.height = height;
        this.frameCount = 0;

        const gl = canvas.getContext('webgl2', { antialias: false });
        if (!gl) throw new Error('WebGL 2 not available');
        this.gl = gl;

        // Required for float FBOs
        const ext = gl.getExtension('EXT_color_buffer_float');
        if (!ext) throw new Error('EXT_color_buffer_float not available');

        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);

        this._initShaders();
        this._initGeometry();
        this._initFBOs();

        // Pre-allocate readback buffer for motion vectors
        this._motionReadback = new Float32Array(width * height * 4);
    }

    _initShaders() {
        const { gl } = this;
        this.sceneProg = createProgram(gl, sceneVert, sceneFrag);
        this.displayProg = createProgram(gl, displayVert, displayFrag);

        // Cache uniform locations — scene
        this.sceneUniforms = {
            u_model:           gl.getUniformLocation(this.sceneProg, 'u_model'),
            u_view_proj:       gl.getUniformLocation(this.sceneProg, 'u_view_proj'),
            u_prev_model:      gl.getUniformLocation(this.sceneProg, 'u_prev_model'),
            u_prev_view_proj:  gl.getUniformLocation(this.sceneProg, 'u_prev_view_proj'),
        };

        // Cache uniform locations — display
        this.displayUniforms = {
            noise_tex:      gl.getUniformLocation(this.displayProg, 'noise_tex'),
            color_tex:      gl.getUniformLocation(this.displayProg, 'color_tex'),
            motion_tex:     gl.getUniformLocation(this.displayProg, 'motion_tex'),
            u_display_mode: gl.getUniformLocation(this.displayProg, 'u_display_mode'),
        };
    }

    _initGeometry() {
        const { gl } = this;

        // Cube VAO
        this.cubeVAO = gl.createVertexArray();
        gl.bindVertexArray(this.cubeVAO);
        const cubeVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVBO);
        gl.bufferData(gl.ARRAY_BUFFER, cubeVertices(), gl.STATIC_DRAW);
        this._setupSceneAttribs();
        gl.bindVertexArray(null);

        // Floor VAO
        this.floorVAO = gl.createVertexArray();
        gl.bindVertexArray(this.floorVAO);
        const floorVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, floorVBO);
        gl.bufferData(gl.ARRAY_BUFFER, floorVertices(), gl.STATIC_DRAW);
        this._setupSceneAttribs();
        gl.bindVertexArray(null);

        // Display quad VAO
        this.displayQuadVAO = gl.createVertexArray();
        gl.bindVertexArray(this.displayQuadVAO);
        const quadVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
        gl.bufferData(gl.ARRAY_BUFFER, quadVertices(), gl.STATIC_DRAW);
        this._setupQuadAttribs(this.displayProg);
        gl.bindVertexArray(null);
    }

    _setupSceneAttribs() {
        const { gl } = this;
        const stride = 9 * 4; // 9 floats * 4 bytes
        const posLoc = gl.getAttribLocation(this.sceneProg, 'in_position');
        const normLoc = gl.getAttribLocation(this.sceneProg, 'in_normal');
        const colLoc = gl.getAttribLocation(this.sceneProg, 'in_color');
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(normLoc);
        gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, stride, 12);
        gl.enableVertexAttribArray(colLoc);
        gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, stride, 24);
    }

    _setupQuadAttribs(prog) {
        const { gl } = this;
        const stride = 4 * 4;
        const posLoc = gl.getAttribLocation(prog, 'in_position');
        const tcLoc = gl.getAttribLocation(prog, 'in_texcoord');
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(tcLoc);
        gl.vertexAttribPointer(tcLoc, 2, gl.FLOAT, false, stride, 8);
    }

    _initFBOs() {
        const { gl, width: W, height: H } = this;

        // --- Scene MRT: color (RGBA8) + motion (RGBA32F) + depth ---
        this.colorTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        // Motion vectors: use RGBA32F (readPixels requires 4 components)
        this.motionTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.motionTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        this.depthRB = gl.createRenderbuffer();
        gl.bindRenderbuffer(gl.RENDERBUFFER, this.depthRB);
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, W, H);

        this.sceneFBO = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFBO);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.colorTex, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.motionTex, 0);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depthRB);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);

        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (status !== gl.FRAMEBUFFER_COMPLETE) {
            throw new Error('Scene FBO incomplete: 0x' + status.toString(16));
        }

        // --- Noise texture for display (uploaded from JS each frame) ---
        this.noiseTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.noiseTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    /**
     * Render cube + floor to the MRT framebuffer.
     * @param {Float32Array} model - 4x4 column-major
     * @param {Float32Array} viewProj - 4x4 column-major
     * @param {Float32Array} prevModel - 4x4 column-major
     * @param {Float32Array} prevViewProj - 4x4 column-major
     */
    renderScene(model, viewProj, prevModel, prevViewProj) {
        const { gl } = this;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFBO);
        gl.viewport(0, 0, this.width, this.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);
        gl.useProgram(this.sceneProg);

        const identity = mat4.create();  // gl-matrix identity

        // Floor (static — identity model, zero motion)
        gl.uniformMatrix4fv(this.sceneUniforms.u_model, false, identity);
        gl.uniformMatrix4fv(this.sceneUniforms.u_view_proj, false, viewProj);
        gl.uniformMatrix4fv(this.sceneUniforms.u_prev_model, false, identity);
        gl.uniformMatrix4fv(this.sceneUniforms.u_prev_view_proj, false, prevViewProj);
        gl.bindVertexArray(this.floorVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Cube
        gl.uniformMatrix4fv(this.sceneUniforms.u_model, false, model);
        gl.uniformMatrix4fv(this.sceneUniforms.u_prev_model, false, prevModel);
        gl.bindVertexArray(this.cubeVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 36);

        gl.bindVertexArray(null);
    }

    /**
     * Read motion vectors from the scene FBO.
     * @returns {Float32Array} [H * W * 2] in (mv_x, mv_y) order, row 0 = bottom
     */
    readMotion() {
        const { gl, width: W, height: H } = this;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFBO);
        // Read from motion attachment (COLOR_ATTACHMENT1)
        gl.readBuffer(gl.COLOR_ATTACHMENT1);
        gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, this._motionReadback);

        // Extract RG channels from RGBA → [H*W*2]
        const out = new Float32Array(W * H * 2);
        for (let i = 0; i < W * H; i++) {
            out[i * 2]     = this._motionReadback[i * 4];      // mv_x
            out[i * 2 + 1] = this._motionReadback[i * 4 + 1];  // mv_y
        }
        return out;
    }

    /**
     * Upload noise data to the noise texture for display.
     * @param {Float32Array} data - [H * W * 4] RGBA float, row 0 = bottom (OpenGL order)
     */
    uploadNoise(data) {
        const { gl, width: W, height: H } = this;
        gl.bindTexture(gl.TEXTURE_2D, this.noiseTex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RGBA, gl.FLOAT, data);
        this.frameCount++;
    }

    /**
     * Render final output to the canvas.
     * @param {number} mode - 0=noise, 1=color, 2=motion, 3=side-by-side, 4=raw
     */
    display(mode = 0) {
        const { gl } = this;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.width, this.height);
        gl.clearColor(0.1, 0.1, 0.1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE);

        gl.useProgram(this.displayProg);

        // Bind textures
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.noiseTex);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, this.motionTex);

        gl.uniform1i(this.displayUniforms.noise_tex, 0);
        gl.uniform1i(this.displayUniforms.color_tex, 1);
        gl.uniform1i(this.displayUniforms.motion_tex, 2);
        gl.uniform1i(this.displayUniforms.u_display_mode, mode);

        gl.bindVertexArray(this.displayQuadVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        gl.bindVertexArray(null);
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);
    }
}
