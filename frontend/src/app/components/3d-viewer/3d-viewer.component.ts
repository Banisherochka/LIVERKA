import { Component, OnInit, OnDestroy, ElementRef, ViewChild, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSliderModule } from '@angular/material/slider';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

/**
 * 3D Viewer Component with Editing Capabilities
 * 
 * Функции:
 * - Визуализация 3D модели печени
 * - Вращение, масштабирование, перемещение камеры
 * - Редактирование модели (коррекция сегментации)
 * - Сглаживание поверхности
 * - Отмена/повтор действий
 * - Экспорт отредактированной модели
 */
@Component({
  selector: 'app-3d-viewer',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatToolbarModule,
    MatTooltipModule,
    MatSliderModule
  ],
  templateUrl: './3d-viewer.component.html',
  styleUrls: ['./3d-viewer.component.scss']
})
export class ThreeDViewerComponent implements OnInit, OnDestroy {
  @ViewChild('rendererContainer', { static: true }) rendererContainer!: ElementRef<HTMLDivElement>;
  @Input() contourData: any;
  @Input() taskId: number | null = null;

  // Three.js objects
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private liverMesh!: THREE.Mesh;
  private animationId?: number;

  // Editing state
  editMode = false;
  showWireframe = false;
  opacity = 0.9;
  
  // History for undo/redo
  private history: any[] = [];
  private historyIndex = -1;

  // UI state
  loading = false;
  error: string | null = null;

  constructor() {}

  ngOnInit() {
    this.initThreeJS();
    this.loadLiverModel();
    this.animate();
  }

  ngOnDestroy() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    if (this.controls) {
      this.controls.dispose();
    }
    if (this.renderer) {
      this.renderer.dispose();
    }
  }

  /**
   * Инициализация Three.js сцены
   */
  private initThreeJS() {
    const container = this.rendererContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a1a);

    // Camera
    this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    this.camera.position.set(0, 0, 300);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    // Controls (OrbitControls для вращения)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 100;
    this.controls.maxDistance = 500;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    this.scene.add(directionalLight);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-10, -10, -10);
    this.scene.add(directionalLight2);

    // Grid helper
    const gridHelper = new THREE.GridHelper(400, 20, 0x444444, 0x222222);
    this.scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(150);
    this.scene.add(axesHelper);

    // Handle window resize
    window.addEventListener('resize', this.onWindowResize.bind(this));
  }

  /**
   * Загрузка и создание 3D модели печени из contour data
   */
  private loadLiverModel() {
    if (!this.contourData || !this.contourData.slices) {
      console.error('No contour data available');
      this.error = 'Нет данных контуров для построения 3D модели';
      return;
    }

    this.loading = true;

    try {
      // Создание геометрии из contour points
      const geometry = this.createGeometryFromContours(this.contourData.slices);

      // Материал с возможностью прозрачности
      const material = new THREE.MeshPhongMaterial({
        color: 0x8B4513,  // Liver color (brown-ish)
        transparent: true,
        opacity: this.opacity,
        side: THREE.DoubleSide,
        flatShading: false,
        shininess: 30
      });

      // Создание mesh
      this.liverMesh = new THREE.Mesh(geometry, material);
      this.scene.add(this.liverMesh);

      // Center the model
      geometry.computeBoundingBox();
      const boundingBox = geometry.boundingBox!;
      const center = new THREE.Vector3();
      boundingBox.getCenter(center);
      this.liverMesh.position.sub(center);

      this.loading = false;
      
      // Save initial state to history
      this.saveState();

    } catch (error) {
      console.error('Error creating 3D model:', error);
      this.error = 'Ошибка создания 3D модели';
      this.loading = false;
    }
  }

  /**
   * Создание Three.js геометрии из contour points
   */
  private createGeometryFromContours(slices: any[]): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();
    const vertices: number[] = [];
    const indices: number[] = [];

    // Преобразование contour points в vertices
    slices.forEach((slice, sliceIndex) => {
      const contourPoints = slice.contour_points || [];
      const z = sliceIndex * 5; // Spacing between slices

      contourPoints.forEach((point: any) => {
        vertices.push(point.x, point.y, z);
      });
    });

    // Создание triangles для поверхности
    for (let i = 0; i < slices.length - 1; i++) {
      const slice1 = slices[i].contour_points || [];
      const slice2 = slices[i + 1].contour_points || [];
      const pointsPerSlice = slice1.length;

      for (let j = 0; j < pointsPerSlice; j++) {
        const next = (j + 1) % pointsPerSlice;
        const base1 = i * pointsPerSlice;
        const base2 = (i + 1) * pointsPerSlice;

        // Triangle 1
        indices.push(base1 + j, base2 + j, base1 + next);
        // Triangle 2
        indices.push(base1 + next, base2 + j, base2 + next);
      }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    return geometry;
  }

  /**
   * Animation loop
   */
  private animate() {
    this.animationId = requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Handle window resize
   */
  private onWindowResize() {
    const container = this.rendererContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  /**
   * Toggle edit mode
   */
  toggleEditMode() {
    this.editMode = !this.editMode;
    if (this.editMode) {
      this.controls.enabled = false; // Disable orbit when editing
    } else {
      this.controls.enabled = true;
    }
  }

  /**
   * Toggle wireframe view
   */
  toggleWireframe() {
    this.showWireframe = !this.showWireframe;
    if (this.liverMesh) {
      (this.liverMesh.material as THREE.MeshPhongMaterial).wireframe = this.showWireframe;
    }
  }

  /**
   * Change opacity
   */
  onOpacityChange(event: any) {
    this.opacity = event.value;
    if (this.liverMesh) {
      (this.liverMesh.material as THREE.MeshPhongMaterial).opacity = this.opacity;
    }
  }

  /**
   * Smooth surface
   */
  smoothSurface() {
    if (!this.liverMesh) return;

    // Apply Laplacian smoothing
    const geometry = this.liverMesh.geometry as THREE.BufferGeometry;
    const positions = geometry.attributes['position'];
    const smoothedPositions = new Float32Array(positions.array.length);

    // Simple averaging for smoothing
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const z = positions.getZ(i);

      // Average with neighbors (simplified)
      smoothedPositions[i * 3] = x * 0.8 + (i > 0 ? positions.getX(i - 1) : x) * 0.1 + 
                                  (i < positions.count - 1 ? positions.getX(i + 1) : x) * 0.1;
      smoothedPositions[i * 3 + 1] = y * 0.8 + (i > 0 ? positions.getY(i - 1) : y) * 0.1 + 
                                      (i < positions.count - 1 ? positions.getY(i + 1) : y) * 0.1;
      smoothedPositions[i * 3 + 2] = z;
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(smoothedPositions, 3));
    geometry.computeVertexNormals();
    geometry.attributes['position'].needsUpdate = true;

    this.saveState();
  }

  /**
   * Reset view to initial state
   */
  resetView() {
    this.camera.position.set(0, 0, 300);
    this.camera.lookAt(0, 0, 0);
    this.controls.reset();
  }

  /**
   * Save current state to history
   */
  private saveState() {
    if (!this.liverMesh) return;

    const state = {
      positions: (this.liverMesh.geometry as THREE.BufferGeometry).attributes['position'].array.slice(),
      opacity: this.opacity,
      wireframe: this.showWireframe
    };

    // Remove states after current index
    this.history = this.history.slice(0, this.historyIndex + 1);
    this.history.push(state);
    this.historyIndex++;

    // Limit history size
    if (this.history.length > 20) {
      this.history.shift();
      this.historyIndex--;
    }
  }

  /**
   * Undo last action
   */
  undo() {
    if (this.historyIndex > 0) {
      this.historyIndex--;
      this.restoreState(this.history[this.historyIndex]);
    }
  }

  /**
   * Redo last undone action
   */
  redo() {
    if (this.historyIndex < this.history.length - 1) {
      this.historyIndex++;
      this.restoreState(this.history[this.historyIndex]);
    }
  }

  /**
   * Restore state from history
   */
  private restoreState(state: any) {
    if (!this.liverMesh) return;

    const geometry = this.liverMesh.geometry as THREE.BufferGeometry;
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(state.positions, 3));
    geometry.computeVertexNormals();
    geometry.attributes['position'].needsUpdate = true;

    this.opacity = state.opacity;
    this.showWireframe = state.wireframe;

    const material = this.liverMesh.material as THREE.MeshPhongMaterial;
    material.opacity = this.opacity;
    material.wireframe = this.showWireframe;
  }

  /**
   * Check if undo is available
   */
  canUndo(): boolean {
    return this.historyIndex > 0;
  }

  /**
   * Check if redo is available
   */
  canRedo(): boolean {
    return this.historyIndex < this.history.length - 1;
  }

  /**
   * Export edited model (stub - would need backend endpoint)
   */
  exportModel() {
    alert('Экспорт модели будет реализован в следующей версии. Модель будет сохранена в формате STL.');
    // TODO: Implement export functionality
    // Convert geometry to STL format and download or send to backend
  }

  /**
   * Download screenshot
   */
  downloadScreenshot() {
    this.renderer.render(this.scene, this.camera);
    const dataURL = this.renderer.domElement.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `liver-3d-model-${this.taskId || 'preview'}.png`;
    link.href = dataURL;
    link.click();
  }
}
