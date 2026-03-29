// Enhanced Abstract Animation with Anime.js

// Define the animation targets
const shapes = document.querySelectorAll('.shape');
const container = document.querySelector('.container');

// Create a main timeline for continuous animation
const mainTl = anime.timeline({
  duration: 3000,
  easing: 'easeInOutQuad',
  loop: true
});

// Create secondary timeline for more complex interactions
const secondaryTl = anime.timeline({
  duration: 4000,
  easing: 'easeInOutSine',
  loop: true
});

// Color palettes for variety
const colorPalettes = [
  ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffe66d', '#ff9ff3'],
  ['#ff9a9e', '#fad0c4', '#fbc2eb', '#a6c1ee', '#fef9c7'],
  ['#a8edea', '#fed6e3', '#ffecd2', '#fcb69f', '#fff5b7'],
  ['#84fab0', '#8fd3f4', '#f5f7fa', '#e4afcf', '#f08a5d']
];

// Initialize shapes with sophisticated starting positions
shapes.forEach((shape, index) => {
  // More sophisticated initial positioning with orbital patterns
  const angle = (index * Math.PI * 2) / shapes.length;
  const radius = 20 + Math.random() * 15; // 20-35 vmin
  const initialX = Math.cos(angle) * radius;
  const initialY = Math.sin(angle) * radius;
  
  const randomScale = 0.3 + Math.random() * 0.7; // 0.3-1.0
  const randomRotate = Math.random() * 360;
  
  // Set initial transform with smoother distribution
  shape.style.transform = `translate(${initialX}vmin, ${initialY}vmin) scale(${randomScale}) rotate(${randomRotate}deg)`;
  
  // Add subtle hover interaction
  shape.addEventListener('mouseenter', () => {
    anime({
      targets: shape,
      scale: [shape.style.transform.includes('scale(') ? parseFloat(shape.style.transform.match(/scale\(([^)]+)/)[1]) : 1, 1.3],
      duration: 600,
      easing: 'easeOutElastic(1, .5)',
      boxShadow: ['0 0 20px rgba(255,255,255,0.2)', '0 0 40px rgba(255,255,255,0.4)']
    });
  });
  
  shape.addEventListener('mouseleave', () => {
    anime({
      targets: shape,
      scale: [1.3, parseFloat(shape.style.transform.match(/scale\(([^)]+)/)[1]) || 0.5 + Math.random() * 0.5],
      duration: 800,
      easing: 'easeInOutQuad',
      boxShadow: ['0 0 40px rgba(255,255,255,0.4)', '0 0 20px rgba(255,255,255,0.2)']
    });
  });
});

// Main animation: Fluid orbital motion with transformations
mainTl.add({
  targets: shapes,
  translateX: [
    { value: () => anime.random(-25, 25) + 'vmin', duration: 1200 },
    { value: () => anime.random(-25, 25) + 'vmin', duration: 1200 }
  ],
  translateY: [
    { value: () => anime.random(-25, 25) + 'vmin', duration: 1200 },
    { value: () => anime.random(-25, 25) + 'vmin', duration: 1200 }
  ],
  scale: [
    { value: () => 0.4 + Math.random() * 0.6, duration: 1000 },
    { value: () => 0.4 + Math.random() * 0.6, duration: 1000 }
  ],
  rotate: [
    { value: () => anime.random(-180, 180), duration: 1500 },
    { value: () => anime.random(-180, 180), duration: 1500 }
  ],
  borderRadius: [
    { value: () => `${Math.random() * 40}%`, duration: 800 },
    { value: () => `${Math.random() * 40}%`, duration: 800 }
  ],
  delay: (el, i) => i * 150,
  easing: 'easeInOutSine'
})
.add({
  targets: shapes,
  backgroundColor: [
    { value: () => colorPalettes[Math.floor(Math.random() * colorPalettes.length)][Math.floor(Math.random() * 5)], duration: 1000 },
    { value: () => colorPalettes[Math.floor(Math.random() * colorPalettes.length)][Math.floor(Math.random() * 5)], duration: 1000 }
  ],
  delay: (el, i) => i * 200,
  easing: 'easeInOutQuad'
}, "-=1000"); // Overlap with previous animation

// Secondary animation: Independent pulsating and drifting
secondaryTl.add({
  targets: shapes,
  translateX: () => anime.random(-30, 30) + 'vmin',
  translateY: () => anime.random(-30, 30) + 'vmin',
  scale: () => 0.5 + Math.random() * 0.8,
  rotate: () => anime.random(-90, 90),
  delay: (el, i) => i * 300 + Math.random() * 500,
  duration: 2000,
  easing: 'easeInOutQuad'
})
.add({
  targets: shapes,
  borderRadius: () => `${Math.random() * 50}%`,
  backgroundColor: () => {
    const palettes = colorPalettes[Math.floor(Math.random() * colorPalettes.length)];
    return palettes[Math.floor(Math.random() * palettes.length)];
  },
  delay: (el, i) => i * 400,
  duration: 1500,
  easing: 'easeInOutSine'
}, "-=800");

// Container animations for depth
const containerTl = anime.timeline({
  loop: true,
  duration: 6000
});

containerTl.add({
  targets: container,
  scale: [1, 1.02, 1],
  rotate: [0, 2, 0],
  duration: 4000,
  easing: 'easeInOutSine'
})
.add({
  targets: container,
  filter: ['hue-rotate(0deg)', 'hue-rotate(15deg)', 'hue-rotate(0deg)'],
  duration: 3000,
  easing: 'linear'
}, "-=2000");

// Add interactive mouse move effect for container
let mouseX = 0;
let mouseY = 0;

document.addEventListener('mousemove', (e) => {
  // Normalize mouse position to -1 to 1 range
  mouseX = (e.clientX / window.innerWidth) * 2 - 1;
  mouseY = (e.clientY / window.innerHeight) * 2 - 1;
});

// Update shapes based on mouse position for subtle interaction
setInterval(() => {
  shapes.forEach((shape, index) => {
    const influence = 0.05; // How much mouse affects movement
    const moveX = mouseX * influence * (index + 1);
    const moveY = mouseY * influence * (index + 1);
    
    const currentTransform = shape.style.transform;
    const scaleMatch = currentTransform.match(/scale\(([^)]+)/);
    const rotateMatch = currentTransform.match(/rotate\(([^)]+)/);
    
    const currentScale = scaleMatch ? parseFloat(scaleMatch[1]) : 1;
    const currentRotate = rotateMatch ? parseFloat(rotateMatch[1]) : 0;
    
    // Apply subtle mouse-driven movement
    shape.style.transform = `translate(${moveX}vmin, ${moveY}vmin) scale(${currentScale}) rotate(${currentRotate}deg)`;
  });
}, 50);

// Add window resize handling for responsive behavior
window.addEventListener('resize', () => {
  // Adjust container size based on window dimensions while maintaining aspect ratio
  const vminValue = Math.min(window.innerWidth, window.innerHeight) * 0.8 / 100;
  container.style.width = `${80 * vminValue}px`;
  container.style.height = `${80 * vminValue}px`;
});

// Initialize container size on load
window.dispatchEvent(new Event('resize'));

// Add subtle glow animation to shapes
const glowTl = anime.timeline({
  loop: true,
  duration: 4000
});

glowTl.add({
  targets: shapes,
  boxShadow: [
    { value: '0 0 20px rgba(255,255,255,0.2)', duration: 1000 },
    { value: '0 0 40px rgba(255,255,255,0.4)', duration: 1000 },
    { value: '0 0 20px rgba(255,255,255,0.2)', duration: 1000 },
    { value: '0 0 10px rgba(255,255,255,0.1)', duration: 1000 }
  ],
  delay: (el, i) => i * 200,
  easing: 'easeInOutQuad'
});