let GL = null;
const { mat4, mat3, vec2, vec3, vec4, quat } = glMatrix;

const pyramidObj = `
# Blender OBJ File: /home/william/Documents/Blender/pyramid2.obj
# www.blender.org

#StarLogoCommand ComputeNormals

mtllib pyramid2.mtl

o o_Plane_Plane_o_Plane_Plane
v -2.79999995232 -2.10000014305 -2.7999997139
v -2.80000090599 -2.10000014305 2.79999923706
v 2.79999947548 -2.09999966621 2.79999995232
v 2.7999997139 -2.09999966621 -2.80000019073
v -5.51132018245e-07 2.87229824066 -1.66892974107e-07
v -2.79467940331 -2.09685015678 2.79822540283
v 2.79467797279 -2.09684967995 2.79822611809
v -5.51339951471e-07 2.86599826813 0.00354754924774
v -2.79909920692 -2.0984005928 -2.79729771614
v -2.7991001606 -2.0984005928 2.7972972393
v -0.00180196762085 2.86909937859 -1.66938988855e-07
v -2.78854846954 -2.09322142601 -2.79618263245
v 2.78854823112 -2.09322094917 -2.79618310928
v -5.49954393136e-07 2.858741045 -0.0076345205307
v 2.7947371006 -2.09065461159 2.78421282768
v 2.79473733902 -2.09065461159 -2.7842130661
v 0.0105241537094 2.85360813141 -1.66713732597e-07
v -2.78626155853 -2.10000014305 -2.78626132011
v -2.78626251221 -2.10000014305 2.78626084328
v 2.7862610817 -2.09999966621 2.78626155853
v 2.78626132011 -2.09999966621 -2.78626179695
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 1.0 1.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 1.0 1.0 0.0
vt 0.0 1.0 0.0
vt 0.0 0.0 0.0
vt 1.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 1.0 1.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 1.0 1.0 0.0
vt 0.0 1.0 0.0
vt 0.0 0.0 0.0
vt 1.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 1.0 1.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 1.0 1.0 0.0
vt 0.0 1.0 0.0
vt 1.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 1.0 1.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 1.0 1.0 0.0
vt 0.0 1.0 0.0
vt 0.0 0.0 0.0
vt 1.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 0.0 0.0
vt 0.0 1.0 0.0
vt 0.0 0.0 0.0
vt 1.0 0.0 0.0
vt 1.0 1.0 0.0
vn -0.323962181807 -0.888874053955 -0.323962181807
vn 0.323962181807 -0.888874053955 -0.323962181807
vn 0.0 -0.490656763315 -0.8713529706
vn 0.0 -0.490656763315 -0.8713529706
vn 0.707106769085 0.0 -0.707106769085
vn -0.323962181807 -0.888874053955 -0.323962181807
vn 0.0 -0.490656763315 -0.8713529706
vn 0.0 -0.490656763315 -0.8713529706
vn 0.323962181807 -0.888874053955 -0.323962181807
vn 0.707106769085 0.0 -0.707106769085
vn 0.0 -0.490656763315 -0.8713529706
vn 0.0 -0.490656763315 -0.8713529706
vn 0.0 -0.490656763315 -0.8713529706
vn 0.0 -0.490656763315 -0.8713529706
vn 0.0 -0.490656763315 -0.8713529706
vn 0.323962181807 -0.888874053955 -0.323962181807
vn 0.323958963156 -0.888865232468 0.323989480734
vn 0.8713529706 -0.490656763315 0.0
vn 0.8713529706 -0.490656763315 0.0
vn 0.323958963156 -0.888865232468 0.323989480734
vn 0.707106769085 0.0 -0.707106769085
vn 0.871339857578 -0.49067991972 0.0
vn 0.8713529706 -0.490656763315 0.0
vn 0.707106769085 0.0 -0.707106769085
vn 0.323962181807 -0.888874053955 -0.323962181807
vn 0.8713529706 -0.490656763315 0.0
vn 0.871339857578 -0.49067991972 0.0
vn 0.8713529706 -0.490656763315 0.0
vn 0.8713529706 -0.490656763315 0.0
vn 0.871339857578 -0.49067991972 0.0
vn 0.323958963156 -0.888865232468 0.323989480734
vn -0.323962181807 -0.888874053955 0.323962181807
vn 0.0 -0.490656763315 0.8713529706
vn 0.0 -0.490656763315 0.8713529706
vn -0.323962181807 -0.888874053955 0.323962181807
vn 0.707106769085 0.0 -0.707106769085
vn 0.0 -0.490656763315 0.8713529706
vn 0.0 -0.490656763315 0.8713529706
vn 0.707106769085 0.0 -0.707106769085
vn 0.323958963156 -0.888865232468 0.323989480734
vn 0.0 -0.490656763315 0.8713529706
vn 0.0 -0.490656763315 0.8713529706
vn 0.0 -0.490656763315 0.8713529706
vn 0.0 -0.490656763315 0.8713529706
vn 0.0 -0.490656763315 0.8713529706
vn -0.323962181807 -0.888874053955 0.323962181807
vn -0.323962181807 -0.888874053955 -0.323962181807
vn -0.8713529706 -0.490656763315 0.0
vn -0.8713529706 -0.490656763315 0.0
vn 0.707106769085 0.0 -0.707106769085
vn -0.323962181807 -0.888874053955 0.323962181807
vn -0.8713529706 -0.490656763315 0.0
vn -0.8713529706 -0.490656763315 0.0
vn -0.323962181807 -0.888874053955 -0.323962181807
vn 0.707106769085 0.0 -0.707106769085
vn -0.8713529706 -0.490656763315 0.0
vn -0.8713529706 -0.490656763315 0.0
vn -0.8713529706 -0.490656763315 0.0
vn -0.8713529706 -0.490656763315 0.0
vn -0.8713529706 -0.490656763315 0.0
vn 0.323958963156 -0.888865232468 0.323989480734
vn 0.323962181807 -0.888874053955 -0.323962181807
vn 0.0 1.0 0.0
vn 0.0 1.0 0.0
vn 0.323962181807 -0.888874053955 -0.323962181807
vn -0.323962181807 -0.888874053955 -0.323962181807
vn 0.0 1.0 0.0
vn 0.0 1.0 0.0
vn -0.323962181807 -0.888874053955 -0.323962181807
vn -0.323962181807 -0.888874053955 0.323962181807
vn 0.0 1.0 0.0
vn 0.0 1.0 0.0
vn -0.323962181807 -0.888874053955 0.323962181807
vn 0.323958963156 -0.888865232468 0.323989480734
vn 0.0 1.0 0.0
vn 0.0 1.0 0.0
vn 0.0 1.0 0.0
vn 0.0 1.0 0.0
vn 0.0 1.0 0.0
vn 0.0 1.0 0.0
f 3/1/1 2/2/2 6/3/3 7/4/4 
f 5/5/5 3/6/6 7/7/7 8/8/8 
f 2/9/9 5/10/10 8/11/11 6/12/12 
f 6/13/13 7/14/14 8/15/15 
f 2/16/16 1/17/17 9/18/18 10/19/19 
f 1/20/20 5/21/21 11/22/22 9/23/23 
f 5/24/24 2/25/25 10/26/26 11/27/27 
f 9/28/28 10/29/29 11/30/30 
f 1/31/31 4/32/32 13/33/33 12/34/34 
f 4/35/35 5/36/36 14/37/37 13/38/38 
f 5/39/39 1/40/40 12/41/41 14/42/42 
f 12/43/43 14/44/44 13/45/45 
f 4/46/46 3/47/47 15/48/48 16/49/49 
f 5/50/50 4/51/51 16/52/52 17/53/53 
f 3/54/54 5/55/55 17/56/56 15/57/57 
f 15/58/58 16/59/59 17/60/60 
f 1/61/61 2/62/62 19/63/63 18/64/64 
f 2/65/65 3/66/66 20/67/67 19/68/68 
f 3/69/69 4/70/70 21/71/71 20/72/72 
f 4/73/73 1/74/74 18/75/75 21/76/76 
f 19/77/77 18/78/78 21/79/79 20/80/80 
`

const _OPAQUE_VS = `#version 300 es
precision highp float;

uniform mat3 normalMatrix;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec4 colour;
in vec2 uv0;

out vec2 vUV0;
out vec4 vColour;

void main(void) {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  vColour = colour;
  vUV0 = uv0;
}
`;

// фрагментный шейдер для окрашивания
// Cэмплирует буфер освещения и умножает его на текстуру диффузии 
// (необработанный цветовой канал 3D-объекта)
const _OPAQUE_FS = `#version 300 es
precision highp float;

uniform sampler2D diffuseTexture;
uniform sampler2D normalTexture;
uniform sampler2D gBuffer_Light;
uniform vec4 resolution;

in vec4 vColour;
in vec2 vUV0;

layout(location = 0) out vec4 out_FragColour;

void main(void) {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 lightSample = texture(gBuffer_Light, uv);
  vec4 albedo = texture(diffuseTexture, vUV0);
  out_FragColour = (albedo * vec4(lightSample.xyz, 1.0) +
      lightSample.w * vec4(0.3, 0.6, 0.1, 0.0));
}
`;

const _QUAD_VS = `#version 300 es
precision highp float;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec2 uv0;

void main(void) {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// фрагментный шейдер для освещения
// В зависимости от типа освещения (directional/point) мы берем текстуры нормали
// и позиции и вычисляем вклад света относительно типа источника света
// ---------------------------------------------------------------------------------
// В данном шейдере используется модель шейдинга Блинна-Фонга 
// https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model
const _QUAD_FS = `#version 300 es
precision highp float;

uniform sampler2D gBuffer_Normal;
uniform sampler2D gBuffer_Position;
uniform vec3 lightColour;

#define _LIGHT_TYPE_POINT

#ifdef _LIGHT_TYPE_DIRECTIONAL
uniform vec3 lightDirection;
#endif

uniform vec3 lightPosition;
uniform vec3 lightAttenuation;
uniform vec3 cameraPosition;
uniform vec4 resolution;

out vec4 out_FragColour;

#define saturate(a) clamp(a, 0.0, 1.0)

float _SmootherStep(float x, float a, float b) {
  x = x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
  return x * (b - a) + a;
}

vec2 _CalculatePhong(vec3 lightDirection, vec3 cameraPosition, vec3 position, vec3 normal) {
  vec3 viewDirection = normalize(cameraPosition - position);
  vec3 H = normalize(lightDirection.xyz + viewDirection);
  float NdotH = dot(normal.xyz, H);
  float specular = saturate(pow(NdotH, 32.0));
  float diffuse = saturate(dot(lightDirection.xyz, normal.xyz));
  return vec2(diffuse, diffuse * specular);
}

vec4 _CalculateLight_Directional(
    vec3 lightDirection, vec3 lightColour, vec3 position, vec3 normal) {
  vec2 lightSample = _CalculatePhong(-lightDirection, cameraPosition, position, normal);
  return vec4(lightSample.x * lightColour, lightSample.y);
}

vec4 _CalculateLight_Point(
    vec3 lightPosition, vec3 lightAttenuation, vec3 lightColour, vec3 position, vec3 normal) {
  vec3 dirToLight = lightPosition - position;
  float lightDistance = length(dirToLight);
  dirToLight = normalize(dirToLight);
  vec2 lightSample = _CalculatePhong(dirToLight, cameraPosition, position, normal);
  float falloff = saturate((lightDistance - lightAttenuation.x) / lightAttenuation.y);
  lightSample *= _SmootherStep(falloff, 1.0, 0.0);
  return vec4(lightSample.x * lightColour, lightSample.y);
}

void main(void) {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 normal = texture(gBuffer_Normal, uv);
  vec4 position = texture(gBuffer_Position, uv);
#ifdef _LIGHT_TYPE_DIRECTIONAL
  vec4 lightSample = _CalculateLight_Directional(
      lightDirection, lightColour, position.xyz, normal.xyz);
#elif defined(_LIGHT_TYPE_POINT)
  vec4 lightSample = _CalculateLight_Point(
      lightPosition, lightAttenuation, lightColour, position.xyz, normal.xyz);
#endif
  out_FragColour = lightSample;
}
`;

const _QUAD_COLOUR_VS = `#version 300 es
precision highp float;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec2 uv0;

void main(void) {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const _QUAD_COLOUR_FS = `#version 300 es
precision highp float;

uniform sampler2D gQuadTexture;
uniform vec4 resolution;

out vec4 out_FragColour;

void main(void) {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  out_FragColour = texture(gQuadTexture, uv);
}
`;

// Вершинный шейдер
// ---------------------------------------------------------------------------------
// Каждый раз при рендеринге фигуры вершинный шейдер запускается 
// для каждой вершины фигуры. Его задача - преобразовать входную 
// вершину из исходной системы координат в систему координат 
// "клипового пространства", используемую WebGL, в которой каждая 
// ось имеет диапазон от -1,0 до 1,0, независимо от соотношения сторон, 
// фактического размера или любых других факторов.
// ---------------------------------------------------------------------------------
// Здесь вершинный шейдер выполняет необходимые преобразования положения вершины - 
// для этого мы определяем матрицы модели, вида и проекции с помощью униформ.
// Затем преобразованная вершина возвращается, сохраненная в специальной переменной,
// предоставляемой GLSL - gl_Position.

const _SIMPLE_VS = `#version 300 es
precision highp float;

uniform mat3 normalMatrix;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec2 uv0;

out vec4 vWSPosition;
out vec3 vNormal;
out vec3 vTangent;
out vec2 vUV0;

void main(void) {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  vNormal = normalize(normalMatrix * normal);
  vTangent = normalize(normalMatrix * tangent);
  vWSPosition = modelMatrix * vec4(position, 1.0);
  vUV0 = uv0;
}
`;

// Фрагментный шейдер
// ---------------------------------------------------------------------------------
// Фрагментный шейдер вызывается один раз для каждого пикселя каждой рисуемой фигуры 
// после того, как вершины фигуры были обработаны вершинным шейдером. Его задача - 
// определить цвет этого пикселя, выясняя, какой тексель (то есть пиксель из текстуры формы) 
// применить к пикселю, получая цвет этого текселя, а затем применяя соответствующее 
// освещение к цвету. Затем цвет возвращается на слой WebGL, дополнительно сохраняясь
// в специальной переменной gl_FragColor. Затем этот цвет отображается на экране
// в правильном положении для соответствующего пикселя фигуры.

const _SIMPLE_FS = `#version 300 es
precision highp float;

uniform sampler2D normalTexture;

in vec4 vWSPosition;
in vec3 vNormal;
in vec3 vTangent;
in vec2 vUV0;

layout(location = 0) out vec4 out_Normals;
layout(location = 1) out vec4 out_Position;

void main(void) {
  vec3 bitangent = normalize(cross(vTangent, vNormal));
  mat3 tbn = mat3(vTangent, bitangent, vNormal);
  vec3 normalSample = normalize(texture(normalTexture, vUV0).xyz * 2.0 - 1.0);
  vec3 vsNormal = normalize(tbn * normalSample);

  out_Normals = vec4(vsNormal, 1.0);
  out_Position = vWSPosition;
}
`;

function* triangulate(elements) {
    if (elements.length <= 3) {
        yield elements;
    }
    else if (elements.length === 4) {
        yield [elements[0], elements[1], elements[2]];
        yield [elements[2], elements[3], elements[0]];
    }
    else {
        for (let i = 1; i < elements.length - 1; i++) {
            yield [elements[0], elements[i], elements[i + 1]];
        }
    }
}

class Shader {
    constructor(vsrc, fsrc, defines) {
        defines = defines || [];

        this._Init(vsrc, fsrc, defines);
    }

    // инициализация вершинного и фрагментного шейдеров
    _Init(vsrc, fsrc, defines) {
        this._defines = defines;

        vsrc = this._ModifySourceWithDefines(vsrc, defines);
        fsrc = this._ModifySourceWithDefines(fsrc, defines);

        this._vsSource = vsrc;
        this._fsSource = fsrc;

        this._vsProgram = this._Load(GL.VERTEX_SHADER, vsrc);
        this._fsProgram = this._Load(GL.FRAGMENT_SHADER, fsrc);

        this._shader = GL.createProgram();
        GL.attachShader(this._shader, this._vsProgram);
        GL.attachShader(this._shader, this._fsProgram);
        GL.linkProgram(this._shader);

        if (!GL.getProgramParameter(this._shader, GL.LINK_STATUS)) {
            return null;
        }

        // атрибуты - получаемые значения
        this.attribs = {
            positions: GL.getAttribLocation(this._shader, 'position'),
            normals: GL.getAttribLocation(this._shader, 'normal'),
            tangents: GL.getAttribLocation(this._shader, 'tangent'),
            uvs: GL.getAttribLocation(this._shader, 'uv0'),
            colours: GL.getAttribLocation(this._shader, 'colour'),
        };
        // униформы - параметры, активно используемые в представлениях шейдеров с GLSL
        this.uniforms = {
            projectionMatrix: {
                type: 'mat4',
                location: GL.getUniformLocation(this._shader, 'projectionMatrix')
            },
            modelViewMatrix: {
                type: 'mat4',
                location: GL.getUniformLocation(this._shader, 'modelViewMatrix'),
            },
            modelMatrix: {
                type: 'mat4',
                location: GL.getUniformLocation(this._shader, 'modelMatrix'),
            },
            normalMatrix: {
                type: 'mat3',
                location: GL.getUniformLocation(this._shader, 'normalMatrix'),
            },
            resolution: {
                type: 'vec4',
                location: GL.getUniformLocation(this._shader, 'resolution'),
            },
            lightColour: {
                type: 'vec3',
                location: GL.getUniformLocation(this._shader, 'lightColour'),
            },
            lightDirection: {
                type: 'vec3',
                location: GL.getUniformLocation(this._shader, 'lightDirection'),
            },
            lightPosition: {
                type: 'vec3',
                location: GL.getUniformLocation(this._shader, 'lightPosition'),
            },
            lightAttenuation: {
                type: 'vec3',
                location: GL.getUniformLocation(this._shader, 'lightAttenuation'),
            },
            cameraPosition: {
                type: 'vec3',
                location: GL.getUniformLocation(this._shader, 'cameraPosition'),
            },
            diffuseTexture: {
                type: 'texture',
                location: GL.getUniformLocation(this._shader, 'diffuseTexture'),
            },
            normalTexture: {
                type: 'texture',
                location: GL.getUniformLocation(this._shader, 'normalTexture'),
            },
            gBuffer_Light: {
                type: 'texture',
                location: GL.getUniformLocation(this._shader, 'gBuffer_Light'),
            },
            gBuffer_Colour: {
                type: 'texture',
                location: GL.getUniformLocation(this._shader, 'gBuffer_Colour'),
            },
            gBuffer_Normal: {
                type: 'texture',
                location: GL.getUniformLocation(this._shader, 'gBuffer_Normal'),
            },
            gBuffer_Position: {
                type: 'texture',
                location: GL.getUniformLocation(this._shader, 'gBuffer_Position'),
            },
            gQuadTexture: {
                type: 'texture',
                location: GL.getUniformLocation(this._shader, 'gQuadTexture'),
            }
        };
    }

    // загрузка шейдера
    _Load(type, source) {
        const shader = GL.createShader(type);

        GL.shaderSource(shader, source);
        GL.compileShader(shader);

        // если ошибка - не подгружаем
        if (!GL.getShaderParameter(shader, GL.COMPILE_STATUS)) {
            console.log(GL.getShaderInfoLog(shader));
            console.log(source);
            GL.deleteShader(shader);
            return null;
        }

        return shader;
    }

    _ModifySourceWithDefines(src, defines) {
        const lines = src.split('\n');

        const defineStrings = defines.map(d => '#define ' + d);

        lines.splice(3, 0, defineStrings);

        return lines.join('\n');
    }

    // связывание используемого шейдера
    Bind() {
        GL.useProgram(this._shader);
    }
}

class ShaderInstance {
    constructor(shader) {
        this._shaderData = shader;
        this._uniforms = {};

        // копируем все значения униформ из шейдера; оставляем пустым поле "значения", мы будем выставлять его позже
        for (let k in shader.uniforms) {
            this._uniforms[k] = {
                location: shader.uniforms[k].location,
                type: shader.uniforms[k].type,
                value: null
            };
        }
        this._attribs = { ...shader.attribs };
    }

    SetMat4(name, m) {
        this._uniforms[name].value = m;
    }

    SetMat3(name, m) {
        this._uniforms[name].value = m;
    }

    SetVec4(name, v) {
        this._uniforms[name].value = v;
    }

    SetVec3(name, v) {
        this._uniforms[name].value = v;
    }

    SetTexture(name, t) {
        this._uniforms[name].value = t;
    }

    // в связывании проходимся по каждой униформе и выставляем им значения
    Bind(constants) {
        this._shaderData.Bind();

        let textureIndex = 0;

        for (let k in this._uniforms) {
            const v = this._uniforms[k];

            let value = constants[k];
            if (v.value) {
                value = v.value;
            }

            if (value && v.location) {
                const t = v.type;

                if (t == 'mat4') {
                    GL.uniformMatrix4fv(v.location, false, value);
                } else if (t == 'mat3') {
                    GL.uniformMatrix3fv(v.location, false, value);
                } else if (t == 'vec4') {
                    GL.uniform4fv(v.location, value);
                } else if (t == 'vec3') {
                    GL.uniform3fv(v.location, value);
                }
                // состояние индекса нужно отслеживать мануально
                else if (t == 'texture') {
                    value.Bind(textureIndex);
                    GL.uniform1i(v.location, textureIndex);
                    textureIndex++;
                }
            }
        }
    }
}

class Texture {
    constructor() { }

    Load(src) {
        this._name = src;
        this._Load(src);
        return this;
    }

    _Load(src) {
        // сначала инициализируем текстуру как простой синий пиксель, пока не загрузилась основная текстура из сорца
        this._texture = GL.createTexture();
        GL.bindTexture(GL.TEXTURE_2D, this._texture);
        GL.texImage2D(GL.TEXTURE_2D, 0, GL.RGBA,
            1, 1, 0, GL.RGBA, GL.UNSIGNED_BYTE,
            new Uint8Array([0, 0, 255, 255]));

        // задаем сорц нашей картинке (текстуре); после того, как она загрузится, осуществляем MIP маппинг (для LOD)
        const img = new Image();
        img.src = src;
        img.onload = () => {
            GL.bindTexture(GL.TEXTURE_2D, this._texture);
            GL.texImage2D(GL.TEXTURE_2D, 0, GL.RGBA, GL.RGBA, GL.UNSIGNED_BYTE, img);
            GL.generateMipmap(GL.TEXTURE_2D);
            GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.LINEAR_MIPMAP_LINEAR);
            GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.LINEAR);
            GL.bindTexture(GL.TEXTURE_2D, null);
        };
    }

    // применить текстуру
    Bind(index) {
        if (!this._texture) {
            return;
        }
        GL.activeTexture(GL.TEXTURE0 + index);
        GL.bindTexture(GL.TEXTURE_2D, this._texture);
    }

    // убрать текстуру
    Unbind() {
        GL.bindTexture(GL.TEXTURE_2D, null);
    }
}

class Mesh {
    // в конструкторе происходит вызов абстрактного метода OnInit
    // этот метод варьируется в зависимости от меша, который мы хотим вывести в мир (кубик/шар и т.д.)
    constructor(objectData, options) {
        this._buffers = {};

        this._OnInit(objectData, options);
    }

    // запись данных в буфер по названию буфера
    _BufferData(info, name) {
        if (name == 'index') {
            info.buffer = GL.createBuffer();
            GL.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, info.buffer);
            GL.bufferData(GL.ELEMENT_ARRAY_BUFFER, new Uint16Array(info.data), GL.STATIC_DRAW);
        } else {
            info.buffer = GL.createBuffer();
            GL.bindBuffer(GL.ARRAY_BUFFER, info.buffer);
            GL.bufferData(GL.ARRAY_BUFFER, new Float32Array(info.data), GL.STATIC_DRAW);
        }

        this._buffers[name] = info;
    }

    // связывание
    Bind(shader) {
        for (let k in this._buffers) {
            if (shader._attribs[k] == -1) {
                continue;
            }

            const b = this._buffers[k];

            if (k == 'index') {
                GL.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, b.buffer);
            } else {
                GL.bindBuffer(GL.ARRAY_BUFFER, b.buffer);
                GL.vertexAttribPointer(shader._attribs[k], b.size, GL.FLOAT, false, 0, 0);
                GL.enableVertexAttribArray(shader._attribs[k]);
            }
        }
    }

    // выводим меш в мир
    Draw() {
        const vertexCount = this._buffers.index.data.length;
        GL.drawElements(GL.TRIANGLES, vertexCount, GL.UNSIGNED_SHORT, 0);
    }

}

// конкретный инстанс объекта (когда хотим создать кучу мешей со своими свойствами)
class MeshInstance {
    constructor(mesh, shaders, shaderParams) {
        this._mesh = mesh;
        this._shaders = shaders;        // нужен свой инстанс шейдеров (для униформ GLSL)

        shaderParams = shaderParams || {};
        for (let sk in shaders) {
            const s = shaders[sk];
            for (let k in shaderParams) {
                s.SetTexture(k, shaderParams[k]);
            }
        }

        this._position = vec3.create();
        this._scale = vec3.fromValues(1, 1, 1);
        this._rotation = quat.create();
    }

    SetPosition(x, y, z) {
        vec3.set(this._position, x, y, z);
    }

    RotateX(rad) {
        quat.rotateX(this._rotation, this._rotation, rad);
    }

    RotateY(rad) {
        quat.rotateY(this._rotation, this._rotation, rad);
    }

    Scale(x, y, z) {
        vec3.set(this._scale, x, y, z);
    }

    // связывание всех параметров и матриц с WebGL для отображения
    Bind(constants, pass) {
        const modelMatrix = mat4.create();
        mat4.fromRotationTranslationScale(
            modelMatrix, this._rotation, this._position, this._scale);

        const viewMatrix = constants['viewMatrix'];
        const modelViewMatrix = mat4.create();
        mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);

        const normalMatrix = mat3.create();
        mat3.fromMat4(normalMatrix, modelMatrix);
        mat3.invert(normalMatrix, normalMatrix);
        mat3.transpose(normalMatrix, normalMatrix);

        const s = this._shaders[pass];

        s.SetMat4('modelViewMatrix', modelViewMatrix);
        s.SetMat4('modelMatrix', modelMatrix);
        s.SetMat3('normalMatrix', normalMatrix);
        s.Bind(constants);

        this._mesh.Bind(s);
    }

    Draw() {
        this._mesh.Draw();
    }
}

// собственно кубик
class Cube extends Mesh {
    constructor(objectData, options) {
        super(objectData, options);
    }

    _OnInit(objectData, options) {
        const positions = [
            // передняя сторона
            -1.0, -1.0, 1.0,
            1.0, -1.0, 1.0,
            1.0, 1.0, 1.0,
            -1.0, 1.0, 1.0,

            // задняя сторона
            -1.0, -1.0, -1.0,
            -1.0, 1.0, -1.0,
            1.0, 1.0, -1.0,
            1.0, -1.0, -1.0,

            // верхняя сторона
            -1.0, 1.0, -1.0,
            -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, -1.0,

            // нижняя сторона
            -1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0, -1.0, 1.0,
            -1.0, -1.0, 1.0,

            // правая сторона
            1.0, -1.0, -1.0,
            1.0, 1.0, -1.0,
            1.0, 1.0, 1.0,
            1.0, -1.0, 1.0,

            // левая сторона
            -1.0, -1.0, -1.0,
            -1.0, -1.0, 1.0,
            -1.0, 1.0, 1.0,
            -1.0, 1.0, -1.0,
        ];

        // координаты на текстуре ("UV mapping")
        const uvs = [
            // передняя сторона
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,

            // задняя сторона
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,

            // верхняя сторона
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,

            // нижняя сторона
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,

            // правая сторона
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,

            // левая сторона
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ];

        // нормали
        const normals = [
            // передняя сторона
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,

            // задняя сторона
            0.0, 0.0, -1.0,
            0.0, 0.0, -1.0,
            0.0, 0.0, -1.0,
            0.0, 0.0, -1.0,

            // верхняя сторона
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,

            // нижняя сторона
            0.0, -1.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, -1.0, 0.0,

            // правая сторона
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,

            // левая сторона
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
        ];

        // тангенсы
        const tangents = [
            // передняя сторона
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,

            // задняя сторона
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,

            // верхняя сторона
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,

            // нижняя сторона
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,

            // правая сторона
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,

            // левая сторона
            0.0, 0.0, -1.0,
            0.0, 0.0, -1.0,
            0.0, 0.0, -1.0,
            0.0, 0.0, -1.0,
        ];

        // индексы
        const indices = [
            0, 1, 2, 0, 2, 3,           // передняя сторона
            4, 5, 6, 4, 6, 7,           // задняя сторона
            8, 9, 10, 8, 10, 11,        // верхняя сторона
            12, 13, 14, 12, 14, 15,     // нижняя сторона
            16, 17, 18, 16, 18, 19,     // правая сторона
            20, 21, 22, 20, 22, 23,     // левая сторона
        ];

        // цвета сторон
        const faceColors = [
            [1.0, 1.0, 1.0, 1.0],    // передняя сторона: белый
            [1.0, 0.0, 0.0, 1.0],    // задняя сторона: красный
            [0.0, 1.0, 0.0, 1.0],    // верхняя сторона: зеленый
            [0.0, 0.0, 1.0, 1.0],    // нижняя сторона: синий
            [1.0, 1.0, 0.0, 1.0],    // правая сторона: желтый
            [1.0, 0.0, 1.0, 1.0],    // левая сторона: фиолетовый
        ];

        // конвертируем массив цветов в таблицу для всех вершин

        let colours = [];

        for (var j = 0; j < faceColors.length; ++j) {
            const c = faceColors[j];

            // для каждого цвета повторяем 4 раза (т.к. вершин 4)
            colours = colours.concat(c, c, c, c);
        }

        // буферизуем все это добро
        this._BufferData({ size: 3, data: positions }, 'positions');
        this._BufferData({ size: 3, data: normals }, 'normals');
        this._BufferData({ size: 3, data: tangents }, 'tangents');
        this._BufferData({ size: 4, data: colours }, 'colours');
        this._BufferData({ size: 2, data: uvs }, 'uvs');
        this._BufferData({ data: indices }, 'index');
    }
}

// простой четырехугольник
class Quad extends Mesh {
    constructor(objectData, options) {
        super(objectData, options);
    }

    _OnInit(objectData, options) {
        const positions = [
            -0.5, -0.5, 1.0,
            0.5, -0.5, 1.0,
            0.5, 0.5, 1.0,
            -0.5, 0.5, 1.0,
        ];

        const normals = [
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
        ];

        const tangents = [
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
        ];

        const uvs = [
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ];

        const indices = [
            0, 1, 2,
            0, 2, 3,
        ];

        this._BufferData({ size: 3, data: positions }, 'positions');
        this._BufferData({ size: 3, data: normals }, 'normals');
        this._BufferData({ size: 3, data: tangents }, 'tangents');
        this._BufferData({ size: 2, data: uvs }, 'uvs');
        this._BufferData({ data: indices }, 'index');
    }
}

class CustomMesh extends Mesh {
    constructor(objectData, options) {
        super(objectData, options);
    }

    _OnInit(objectData, options) {
        this._buffers = {};
        this.name = "";
        this.indicesPerMaterial = [];
        this.materialsByIndex = {};
        this.tangents = [];
        this.bitangents = [];
        options = options || {};
        options.materials = options.materials || {};
        options.enableWTextureCoord = !!options.enableWTextureCoord;
        // the list of unique vertex, normal, texture, attributes
        this.vertexNormals = [];
        this.textures = [];
        // the indicies to draw the faces
        this.indices = [];
        this.textureStride = options.enableWTextureCoord ? 3 : 2;

        const verts = [];
        const vertNormals = [];
        const textures = [];
        const materialNamesByIndex = [];
        const materialIndicesByName = {};
        // keep track of what material we've seen last
        let currentMaterialIndex = -1;
        let currentObjectByMaterialIndex = 0;
        // unpacking stuff
        const unpacked = {
            verts: [],
            norms: [],
            textures: [],
            hashindices: {},
            indices: [[]],
            materialIndices: [],
            index: 0,
        };
        const VERTEX_RE = /^v\s/;
        const NORMAL_RE = /^vn\s/;
        const TEXTURE_RE = /^vt\s/;
        const FACE_RE = /^f\s/;
        const WHITESPACE_RE = /\s+/;
        const USE_MATERIAL_RE = /^usemtl/;
        // array of lines separated by the newline
        const lines = objectData.split("\n");
        for (let line of lines) {
            line = line.trim();
            if (!line || line.startsWith("#")) {
                continue;
            }
            const elements = line.split(WHITESPACE_RE);
            elements.shift();
            if (VERTEX_RE.test(line)) {
                // if this is a vertex
                verts.push(...elements);
            }
            else if (NORMAL_RE.test(line)) {
                // if this is a vertex normal
                vertNormals.push(...elements);
            }
            else if (TEXTURE_RE.test(line)) {
                let coords = elements;
                // by default, the loader will only look at the U and V
                // coordinates of the vt declaration. So, this truncates the
                // elements to only those 2 values. If W texture coordinate
                // support is enabled, then the texture coordinate is
                // expected to have three values in it.
                if (elements.length > 2 && !options.enableWTextureCoord) {
                    coords = elements.slice(0, 2);
                }
                else if (elements.length === 2 && options.enableWTextureCoord) {
                    // If for some reason W texture coordinate support is enabled
                    // and only the U and V coordinates are given, then we supply
                    // the default value of 0 so that the stride length is correct
                    // when the textures are unpacked below.
                    coords.push("0");
                }
                textures.push(...coords);
            }
            else if (USE_MATERIAL_RE.test(line)) {
                const materialName = elements[0];
                // check to see if we've ever seen it before
                if (!(materialName in materialIndicesByName)) {
                    // new material we've never seen
                    materialNamesByIndex.push(materialName);
                    materialIndicesByName[materialName] = materialNamesByIndex.length - 1;
                    // push new array into indices
                    // already contains an array at index zero, don't add
                    if (materialIndicesByName[materialName] > 0) {
                        unpacked.indices.push([]);
                    }
                }
                // keep track of the current material index
                currentMaterialIndex = materialIndicesByName[materialName];
                // update current index array
                currentObjectByMaterialIndex = currentMaterialIndex;
            }
            else if (FACE_RE.test(line)) {
                // if this is a face
                /*
                split this face into an array of Vertex groups
                for example:
                   f 16/92/11 14/101/22 1/69/1
                becomes:
                  ['16/92/11', '14/101/22', '1/69/1'];
                */
                const triangles = triangulate(elements);
                for (const triangle of triangles) {
                    for (let j = 0, eleLen = triangle.length; j < eleLen; j++) {
                        const hash = triangle[j] + "," + currentMaterialIndex;
                        if (hash in unpacked.hashindices) {
                            unpacked.indices[currentObjectByMaterialIndex].push(unpacked.hashindices[hash]);
                        }
                        else {
                            /*
                        Each element of the face line array is a Vertex which has its
                        attributes delimited by a forward slash. This will separate
                        each attribute into another array:
                            '19/92/11'
                        becomes:
                            Vertex = ['19', '92', '11'];
                        where
                            Vertex[0] is the vertex index
                            Vertex[1] is the texture index
                            Vertex[2] is the normal index
                         Think of faces having Vertices which are comprised of the
                         attributes location (v), texture (vt), and normal (vn).
                         */
                            const vertex = triangle[j].split("/");
                            // it's possible for faces to only specify the vertex
                            // and the normal. In this case, vertex will only have
                            // a length of 2 and not 3 and the normal will be the
                            // second item in the list with an index of 1.
                            const normalIndex = vertex.length - 1;
                            // Vertex position
                            unpacked.verts.push(+verts[(+vertex[0] - 1) * 3 + 0]);
                            unpacked.verts.push(+verts[(+vertex[0] - 1) * 3 + 1]);
                            unpacked.verts.push(+verts[(+vertex[0] - 1) * 3 + 2]);
                            // Vertex textures
                            if (textures.length) {
                                const stride = options.enableWTextureCoord ? 3 : 2;
                                unpacked.textures.push(+textures[(+vertex[1] - 1) * stride + 0]);
                                unpacked.textures.push(+textures[(+vertex[1] - 1) * stride + 1]);
                                if (options.enableWTextureCoord) {
                                    unpacked.textures.push(+textures[(+vertex[1] - 1) * stride + 2]);
                                }
                            }
                            // Vertex normals
                            unpacked.norms.push(+vertNormals[(+vertex[normalIndex] - 1) * 3 + 0]);
                            unpacked.norms.push(+vertNormals[(+vertex[normalIndex] - 1) * 3 + 1]);
                            unpacked.norms.push(+vertNormals[(+vertex[normalIndex] - 1) * 3 + 2]);
                            // Vertex material indices
                            unpacked.materialIndices.push(currentMaterialIndex);
                            // add the newly created Vertex to the list of indices
                            unpacked.hashindices[hash] = unpacked.index;
                            unpacked.indices[currentObjectByMaterialIndex].push(unpacked.hashindices[hash]);
                            // increment the counter
                            unpacked.index += 1;
                        }
                    }
                }
            }
        }
        this.vertices = unpacked.verts;
        this.vertexNormals = unpacked.norms;
        this.textures = unpacked.textures;
        this.vertexMaterialIndices = unpacked.materialIndices;
        this.indices = unpacked.indices[currentObjectByMaterialIndex];
        this.indicesPerMaterial = unpacked.indices;
        this.materialNames = materialNamesByIndex;
        this.materialIndices = materialIndicesByName;
        this.materialsByIndex = {};
        this.calculateTangentsAndBitangents();

        this._BufferData({ size: 3, data: this.vertices }, 'positions');
        this._BufferData({ size: 3, data: this.vertexNormals }, 'normals');
        this._BufferData({ size: 3, data: this.tangents }, 'tangents');
        this._BufferData({ size: 2, data: this.textures }, 'uvs');
        this._BufferData({ data: this.indices }, 'index');
    }
    
    calculateTangentsAndBitangents() {
    //    console.log(this.vertices);
    //    console.log(this.vertices.length);
    //    console.log(this.vertexNormals);
    //    console.log(this.vertexNormals.length);
    //    console.log(this.textures);
    //    console.log(this.textures.length);
        console.assert(!!(this.vertices &&
            this.vertices.length &&
            this.vertexNormals &&
            this.vertexNormals.length &&
            this.textures &&
            this.textures.length), "Missing attributes for calculating tangents and bitangents");
        const unpacked = {
            tangents: [...new Array(this.vertices.length)].map(_ => 0),
            bitangents: [...new Array(this.vertices.length)].map(_ => 0),
        };
        // Loop through all faces in the whole mesh
        const indices = this.indices;
        const vertices = this.vertices;
        const normals = this.vertexNormals;
        const uvs = this.textures;
        for (let i = 0; i < indices.length; i += 3) {
            const i0 = indices[i + 0];
            const i1 = indices[i + 1];
            const i2 = indices[i + 2];
            const x_v0 = vertices[i0 * 3 + 0];
            const y_v0 = vertices[i0 * 3 + 1];
            const z_v0 = vertices[i0 * 3 + 2];
            const x_uv0 = uvs[i0 * 2 + 0];
            const y_uv0 = uvs[i0 * 2 + 1];
            const x_v1 = vertices[i1 * 3 + 0];
            const y_v1 = vertices[i1 * 3 + 1];
            const z_v1 = vertices[i1 * 3 + 2];
            const x_uv1 = uvs[i1 * 2 + 0];
            const y_uv1 = uvs[i1 * 2 + 1];
            const x_v2 = vertices[i2 * 3 + 0];
            const y_v2 = vertices[i2 * 3 + 1];
            const z_v2 = vertices[i2 * 3 + 2];
            const x_uv2 = uvs[i2 * 2 + 0];
            const y_uv2 = uvs[i2 * 2 + 1];
            const x_deltaPos1 = x_v1 - x_v0;
            const y_deltaPos1 = y_v1 - y_v0;
            const z_deltaPos1 = z_v1 - z_v0;
            const x_deltaPos2 = x_v2 - x_v0;
            const y_deltaPos2 = y_v2 - y_v0;
            const z_deltaPos2 = z_v2 - z_v0;
            const x_uvDeltaPos1 = x_uv1 - x_uv0;
            const y_uvDeltaPos1 = y_uv1 - y_uv0;
            const x_uvDeltaPos2 = x_uv2 - x_uv0;
            const y_uvDeltaPos2 = y_uv2 - y_uv0;
            const rInv = x_uvDeltaPos1 * y_uvDeltaPos2 - y_uvDeltaPos1 * x_uvDeltaPos2;
            const r = 1.0 / Math.abs(rInv < 0.0001 ? 1.0 : rInv);
            // Tangent
            const x_tangent = (x_deltaPos1 * y_uvDeltaPos2 - x_deltaPos2 * y_uvDeltaPos1) * r;
            const y_tangent = (y_deltaPos1 * y_uvDeltaPos2 - y_deltaPos2 * y_uvDeltaPos1) * r;
            const z_tangent = (z_deltaPos1 * y_uvDeltaPos2 - z_deltaPos2 * y_uvDeltaPos1) * r;
            // Bitangent
            const x_bitangent = (x_deltaPos2 * x_uvDeltaPos1 - x_deltaPos1 * x_uvDeltaPos2) * r;
            const y_bitangent = (y_deltaPos2 * x_uvDeltaPos1 - y_deltaPos1 * x_uvDeltaPos2) * r;
            const z_bitangent = (z_deltaPos2 * x_uvDeltaPos1 - z_deltaPos1 * x_uvDeltaPos2) * r;
            // Gram-Schmidt orthogonalize
            //t = glm::normalize(t - n * glm:: dot(n, t));
            const x_n0 = normals[i0 * 3 + 0];
            const y_n0 = normals[i0 * 3 + 1];
            const z_n0 = normals[i0 * 3 + 2];
            const x_n1 = normals[i1 * 3 + 0];
            const y_n1 = normals[i1 * 3 + 1];
            const z_n1 = normals[i1 * 3 + 2];
            const x_n2 = normals[i2 * 3 + 0];
            const y_n2 = normals[i2 * 3 + 1];
            const z_n2 = normals[i2 * 3 + 2];
            // Tangent
            const n0_dot_t = x_tangent * x_n0 + y_tangent * y_n0 + z_tangent * z_n0;
            const n1_dot_t = x_tangent * x_n1 + y_tangent * y_n1 + z_tangent * z_n1;
            const n2_dot_t = x_tangent * x_n2 + y_tangent * y_n2 + z_tangent * z_n2;
            const x_resTangent0 = x_tangent - x_n0 * n0_dot_t;
            const y_resTangent0 = y_tangent - y_n0 * n0_dot_t;
            const z_resTangent0 = z_tangent - z_n0 * n0_dot_t;
            const x_resTangent1 = x_tangent - x_n1 * n1_dot_t;
            const y_resTangent1 = y_tangent - y_n1 * n1_dot_t;
            const z_resTangent1 = z_tangent - z_n1 * n1_dot_t;
            const x_resTangent2 = x_tangent - x_n2 * n2_dot_t;
            const y_resTangent2 = y_tangent - y_n2 * n2_dot_t;
            const z_resTangent2 = z_tangent - z_n2 * n2_dot_t;
            const magTangent0 = Math.sqrt(x_resTangent0 * x_resTangent0 + y_resTangent0 * y_resTangent0 + z_resTangent0 * z_resTangent0);
            const magTangent1 = Math.sqrt(x_resTangent1 * x_resTangent1 + y_resTangent1 * y_resTangent1 + z_resTangent1 * z_resTangent1);
            const magTangent2 = Math.sqrt(x_resTangent2 * x_resTangent2 + y_resTangent2 * y_resTangent2 + z_resTangent2 * z_resTangent2);
            // Bitangent
            const n0_dot_bt = x_bitangent * x_n0 + y_bitangent * y_n0 + z_bitangent * z_n0;
            const n1_dot_bt = x_bitangent * x_n1 + y_bitangent * y_n1 + z_bitangent * z_n1;
            const n2_dot_bt = x_bitangent * x_n2 + y_bitangent * y_n2 + z_bitangent * z_n2;
            const x_resBitangent0 = x_bitangent - x_n0 * n0_dot_bt;
            const y_resBitangent0 = y_bitangent - y_n0 * n0_dot_bt;
            const z_resBitangent0 = z_bitangent - z_n0 * n0_dot_bt;
            const x_resBitangent1 = x_bitangent - x_n1 * n1_dot_bt;
            const y_resBitangent1 = y_bitangent - y_n1 * n1_dot_bt;
            const z_resBitangent1 = z_bitangent - z_n1 * n1_dot_bt;
            const x_resBitangent2 = x_bitangent - x_n2 * n2_dot_bt;
            const y_resBitangent2 = y_bitangent - y_n2 * n2_dot_bt;
            const z_resBitangent2 = z_bitangent - z_n2 * n2_dot_bt;
            const magBitangent0 = Math.sqrt(x_resBitangent0 * x_resBitangent0 +
                y_resBitangent0 * y_resBitangent0 +
                z_resBitangent0 * z_resBitangent0);
            const magBitangent1 = Math.sqrt(x_resBitangent1 * x_resBitangent1 +
                y_resBitangent1 * y_resBitangent1 +
                z_resBitangent1 * z_resBitangent1);
            const magBitangent2 = Math.sqrt(x_resBitangent2 * x_resBitangent2 +
                y_resBitangent2 * y_resBitangent2 +
                z_resBitangent2 * z_resBitangent2);
            unpacked.tangents[i0 * 3 + 0] += x_resTangent0 / magTangent0;
            unpacked.tangents[i0 * 3 + 1] += y_resTangent0 / magTangent0;
            unpacked.tangents[i0 * 3 + 2] += z_resTangent0 / magTangent0;
            unpacked.tangents[i1 * 3 + 0] += x_resTangent1 / magTangent1;
            unpacked.tangents[i1 * 3 + 1] += y_resTangent1 / magTangent1;
            unpacked.tangents[i1 * 3 + 2] += z_resTangent1 / magTangent1;
            unpacked.tangents[i2 * 3 + 0] += x_resTangent2 / magTangent2;
            unpacked.tangents[i2 * 3 + 1] += y_resTangent2 / magTangent2;
            unpacked.tangents[i2 * 3 + 2] += z_resTangent2 / magTangent2;
            unpacked.bitangents[i0 * 3 + 0] += x_resBitangent0 / magBitangent0;
            unpacked.bitangents[i0 * 3 + 1] += y_resBitangent0 / magBitangent0;
            unpacked.bitangents[i0 * 3 + 2] += z_resBitangent0 / magBitangent0;
            unpacked.bitangents[i1 * 3 + 0] += x_resBitangent1 / magBitangent1;
            unpacked.bitangents[i1 * 3 + 1] += y_resBitangent1 / magBitangent1;
            unpacked.bitangents[i1 * 3 + 2] += z_resBitangent1 / magBitangent1;
            unpacked.bitangents[i2 * 3 + 0] += x_resBitangent2 / magBitangent2;
            unpacked.bitangents[i2 * 3 + 1] += y_resBitangent2 / magBitangent2;
            unpacked.bitangents[i2 * 3 + 2] += z_resBitangent2 / magBitangent2;
            // TODO: check handedness
        }
        this.tangents = unpacked.tangents;
        this.bitangents = unpacked.bitangents;
    }
}

// Родительский класс для всех видов камер
class Camera {
    constructor() {
        this._position = vec3.create();
        this._target = vec3.create();
        this._viewMatrix = mat4.create();
        this._cameraMatrix = mat4.create();
    }

    // задаем расположение камеры
    SetPosition(x, y, z) {
        vec3.set(this._position, x, y, z);
    }

    // задаем направление камеры
    SetTarget(x, y, z) {
        vec3.set(this._target, x, y, z);
    }

    UpdateConstants(constants) {
        mat4.lookAt(this._viewMatrix, this._position, this._target, vec3.fromValues(0, 1, 0));
        mat4.invert(this._cameraMatrix, this._viewMatrix);

        constants['projectionMatrix'] = this._projectionMatrix;
        constants['viewMatrix'] = this._viewMatrix;
        constants['cameraMatrix'] = this._cameraMatrix;
        constants['cameraPosition'] = this._position;
    }
}

// Камера с перспективной проекцией
// Этот режим проекции имитирует то, что видит человеческий глаз.
class PerspectiveCamera extends Camera {
    constructor(fov, aspect, zNear, zFar) {
        super();

        this._projectionMatrix = mat4.create();
        this._fov = fov;
        this._aspect = aspect;
        this._zNear = zNear;
        this._zFar = zFar;

        mat4.perspective(this._projectionMatrix, fov * Math.PI / 180.0, aspect, zNear, zFar);
    }

    GetUp() {
        const v = vec4.fromValues(0, 0, 1, 0);

        vec4.transformMat4(v, v, this._cameraMatrix);

        return v;
    }

    GetRight() {
        const v = vec4.fromValues(1, 0, 0, 0);

        vec4.transformMat4(v, v, this._cameraMatrix);

        return v;
    }
}

// Ортографическая камера
// В этом режиме проецирования размер объекта на визуализированном
// изображении остается постоянным независимо от расстояния до камеры.
class OrthoCamera extends Camera {
    constructor(l, r, b, t, n, f) {
        super();

        this._projectionMatrix = mat4.create();

        mat4.ortho(this._projectionMatrix, l, r, b, t, n, f);
    }
}

// Абстрактный класс для определения типов источников света
class Light {
    constructor() {
    }

    UpdateConstants() {
    }
}

// направленный свет: имитация отдаленных источников света в некотором направлении
class DirectionalLight extends Light {
    constructor() {
        super();

        this._colour = vec3.fromValues(1, 1, 1);
        this._direction = vec3.fromValues(1, 0, 0);
    }

    get Type() {
        return 'Directional';
    }

    // задаем цвет
    SetColour(r, g, b) {
        vec3.set(this._colour, r, g, b);
    }

    // задаем направление
    SetDirection(x, y, z) {
        vec3.set(this._direction, x, y, z);
        vec3.normalize(this._direction, this._direction);
    }

    UpdateConstants(constants) {
        constants['lightDirection'] = this._direction;
        constants['lightColour'] = this._colour;
    }
}

// свет-точка: источник света, распространяющий освещение во всех направлениях от данной точки
class PointLight extends Light {
    constructor() {
        super();

        this._colour = vec3.fromValues(1, 1, 1);
        this._position = vec3.create();
        this._attenuation = vec3.create();
    }

    get Type() {
        return 'Point';
    }

    // задаем цвет
    SetColour(r, g, b) {
        vec3.set(this._colour, r, g, b);
    }

    // задаем расположение
    SetPosition(x, y, z) {
        vec3.set(this._position, x, y, z);
    }

    // задаем радиус распространения
    SetRadius(r1, r2) {
        vec3.set(this._attenuation, r1, r2, 0);
    }

    UpdateConstants(constants) {
        constants['lightPosition'] = this._position;
        constants['lightColour'] = this._colour;
        constants['lightAttenuation'] = this._attenuation;
    }
}

// Класс, осуществляющий основную работу по рендерингу
class Renderer {
    constructor() {
        this._Init();
    }

    _Init() {
        this._canvas = document.createElement('canvas');

        document.body.appendChild(this._canvas);

        // инициализируем контекст WebGL
        GL = this._canvas.getContext('webgl2');

        // продолжаем только в том случае, если текущий браузер поддерживает WebGL
        if (GL === null) {
            alert('Unable to initialize WebGL. Your browser or machine may not support it.');
            return;
        }

        this._constants = {};

        // загружаем сорцы текстур
        this._textures = {};
        this._textures['test-diffuse'] = new Texture().Load('./resources/rough-wet-cobble-albedo-1024.png');
        this._textures['test-normal'] = new Texture().Load('./resources/rough-wet-cobble-normal-1024.jpg');
        this._textures['worn-bumpy-rock-albedo'] = new Texture().Load(
            './resources/worn-bumpy-rock-albedo-1024.png');
        this._textures['worn-bumpy-rock-normal'] = new Texture().Load(
            './resources/worn-bumpy-rock-normal-1024.jpg');

        // инициализируем шейдеры
        this._shaders = {};
        this._shaders['z'] = new Shader(_SIMPLE_VS, _SIMPLE_FS);
        this._shaders['default'] = new Shader(_OPAQUE_VS, _OPAQUE_FS);

        this._shaders['post-quad-colour'] = new Shader(
            _QUAD_COLOUR_VS, _QUAD_COLOUR_FS);
        this._shaders['post-quad-directional'] = new Shader(
            _QUAD_VS, _QUAD_FS, ['_LIGHT_TYPE_DIRECTIONAL']);
        this._shaders['post-quad-point'] = new Shader(
            _QUAD_VS, _QUAD_FS, ['_LIGHT_TYPE_POINT']);

        // инициализируем камеру
        this._camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1.0, 1000.0);
        this._camera.SetPosition(0, 30, 20);
        this._camera.SetTarget(0, 0, -20);

        this._postCamera = new OrthoCamera(0.0, 1.0, 0.0, 1.0, 1.0, 1000.0);

        this._meshes = [];
        this._lights = [];

        // инициализируем "абстрактные" источники света
        this._quadDirectional = new MeshInstance(
            new Quad(),
            { light: new ShaderInstance(this._shaders['post-quad-directional']) });
        this._quadDirectional.SetPosition(0.5, 0.5, -10.0);

        this._quadPoint = new MeshInstance(
            new Quad(),
            { light: new ShaderInstance(this._shaders['post-quad-point']) });
        this._quadPoint.SetPosition(0.5, 0.5, -10.0);

        this._quadColour = new MeshInstance(
            new Quad(),
            { colour: new ShaderInstance(this._shaders['post-quad-colour']) });
        this._quadColour.SetPosition(0.5, 0.5, -10.0);

        this._InitGBuffer();
        this.Resize(window.innerWidth, window.innerHeight);
    }

    _InitGBuffer() {
        // для текстур с плавающей точкой (используется для повышенной точности по сравнению с целочисленными)
        GL.getExtension('EXT_color_buffer_float');

        // буфер глубины (z-буфер)
        this._depthBuffer = GL.createRenderbuffer();
        GL.bindRenderbuffer(GL.RENDERBUFFER, this._depthBuffer);
        GL.renderbufferStorage(
            GL.RENDERBUFFER,
            GL.DEPTH_COMPONENT24,
            window.innerWidth, window.innerHeight);
        GL.bindRenderbuffer(GL.RENDERBUFFER, null);

        // буфер нормали
        this._normalBuffer = GL.createTexture();
        GL.bindTexture(GL.TEXTURE_2D, this._normalBuffer);
        GL.texImage2D(
            GL.TEXTURE_2D, 0, GL.RGBA32F, window.innerWidth, window.innerHeight,
            0, GL.RGBA, GL.FLOAT, null);
        GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
        GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
        GL.bindTexture(GL.TEXTURE_2D, null);

        // буфер положения в мировом пространстве (для фрагментного шейдера)
        this._positionBuffer = GL.createTexture();
        GL.bindTexture(GL.TEXTURE_2D, this._positionBuffer);
        GL.texImage2D(
            GL.TEXTURE_2D, 0, GL.RGBA32F, window.innerWidth, window.innerHeight,
            0, GL.RGBA, GL.FLOAT, null);
        GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
        GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
        GL.bindTexture(GL.TEXTURE_2D, null);

        // буфер освещения
        this._lightBuffer = GL.createTexture();
        GL.bindTexture(GL.TEXTURE_2D, this._lightBuffer);
        GL.texImage2D(
            GL.TEXTURE_2D, 0, GL.RGBA32F, window.innerWidth, window.innerHeight,
            0, GL.RGBA, GL.FLOAT, null);
        GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
        GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
        GL.bindTexture(GL.TEXTURE_2D, null);

        // буфер цвета
        this._colourBuffer = GL.createTexture();
        GL.bindTexture(GL.TEXTURE_2D, this._colourBuffer);
        GL.texImage2D(
            GL.TEXTURE_2D, 0, GL.RGBA32F, window.innerWidth, window.innerHeight,
            0, GL.RGBA, GL.FLOAT, null);
        GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
        GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
        GL.bindTexture(GL.TEXTURE_2D, null);

        // создаем FBO из z-буфера для рендеринга, также связываем буфер нормали и буфер положения
        this._zFBO = GL.createFramebuffer();
        GL.bindFramebuffer(GL.FRAMEBUFFER, this._zFBO);
        GL.framebufferRenderbuffer(
            GL.FRAMEBUFFER, GL.DEPTH_ATTACHMENT, GL.RENDERBUFFER, this._depthBuffer);
        GL.framebufferTexture2D(
            GL.DRAW_FRAMEBUFFER, GL.COLOR_ATTACHMENT0, GL.TEXTURE_2D, this._normalBuffer, 0);
        GL.framebufferTexture2D(
            GL.DRAW_FRAMEBUFFER, GL.COLOR_ATTACHMENT1, GL.TEXTURE_2D, this._positionBuffer, 0);
        GL.bindFramebuffer(GL.FRAMEBUFFER, null);

        // создаем FBO из буфера освещения для рендеринга
        this._lightFBO = GL.createFramebuffer();
        GL.bindFramebuffer(GL.FRAMEBUFFER, this._lightFBO);
        GL.framebufferRenderbuffer(
            GL.FRAMEBUFFER, GL.DEPTH_ATTACHMENT, GL.RENDERBUFFER, this._depthBuffer);
        GL.framebufferTexture2D(
            GL.DRAW_FRAMEBUFFER, GL.COLOR_ATTACHMENT0, GL.TEXTURE_2D, this._lightBuffer, 0);
        GL.bindFramebuffer(GL.FRAMEBUFFER, null);

        // создаем FBO из буфера цвета для рендеринга 
        // (в принципе можно было бы напрямую работать с фрейм буфером, но так можно добавлять пост-обработку и прочее)
        this._colourFBO = GL.createFramebuffer();
        GL.bindFramebuffer(GL.FRAMEBUFFER, this._colourFBO);
        GL.framebufferRenderbuffer(
            GL.FRAMEBUFFER, GL.DEPTH_ATTACHMENT, GL.RENDERBUFFER, this._depthBuffer);
        GL.framebufferTexture2D(
            GL.DRAW_FRAMEBUFFER, GL.COLOR_ATTACHMENT0, GL.TEXTURE_2D, this._colourBuffer, 0);
        GL.bindFramebuffer(GL.FRAMEBUFFER, null);

        this._normalTexture = new Texture();
        this._normalTexture._texture = this._normalBuffer;

        this._positionTexture = new Texture();
        this._positionTexture._texture = this._positionBuffer;

        this._lightTexture = new Texture();
        this._lightTexture._texture = this._lightBuffer;

        this._colourTexture = new Texture();
        this._colourTexture._texture = this._colourBuffer;
    }

    // метод для создания меш инстанса с шейдерами
    CreateMeshInstance(mesh, shaderParams) {
        const params = {};
        for (let k in shaderParams.params) {
            params[k] = this._textures[shaderParams.params[k]];
        }

        const m = new MeshInstance(
            mesh,
            {
                z: new ShaderInstance(this._shaders['z']),
                colour: new ShaderInstance(this._shaders[shaderParams.shader])
            }, params);

        this._meshes.push(m);

        return m;
    }

    // метод для создания источника света определенного типа
    CreateLight(type) {
        let l = null;

        if (type == 'directional') {
            l = new DirectionalLight();
        } else if (type == 'point') {
            l = new PointLight();
        }

        if (!l) {
            return null;
        }

        this._lights.push(l);

        return l;
    }

    // изменение размера полотна
    Resize(w, h) {
        this._canvas.width = w;
        this._canvas.height = h;
        GL.viewport(0, 0, w, h);
    }

    // для рендеринга только определенной части (четырехугольника) рабочего пространства, 
    // которое мы ВИДИМ - пространство, ничем не освещающееся, не рендерится
    _SetQuadSizeForLight(quad, light) {
        const w = mat4.create();
        mat4.fromTranslation(w, light._position);

        const viewMatrix = this._camera._viewMatrix;
        const projectionMatrix = this._camera._projectionMatrix;

        const _TransformToScreenSpace = (p) => {
            const screenPos = vec4.fromValues(
                p[0], p[1], p[2], 1.0);

            vec4.transformMat4(screenPos, screenPos, projectionMatrix);

            screenPos[0] = (screenPos[0] / screenPos[3]) * 0.5 + 0.5;
            screenPos[1] = (screenPos[1] / screenPos[3]) * 0.5 + 0.5;

            return screenPos;
        };

        const lightRadius = (light._attenuation[0] + light._attenuation[1]);
        const lightDistance = vec3.distance(this._camera._position, light._position);

        if (lightDistance < lightRadius) {
            quad.SetPosition(0.5, 0.5, -10);
            quad.Scale(1, 1, 1);
        } else {
            const viewSpaceCenter = vec3.clone(light._position);
            vec3.transformMat4(viewSpaceCenter, viewSpaceCenter, viewMatrix);

            const rightPos = vec3.clone(viewSpaceCenter);
            const upPos = vec3.clone(viewSpaceCenter);
            vec3.add(rightPos, rightPos, vec3.fromValues(lightRadius, 0, 0));
            vec3.add(upPos, upPos, vec3.fromValues(0, -lightRadius, 0));

            const center = _TransformToScreenSpace(light._position);
            const up = _TransformToScreenSpace(upPos);
            const right = _TransformToScreenSpace(rightPos);

            const radius = 2 * Math.max(
                vec2.distance(center, up), vec2.distance(center, right));

            quad.SetPosition(center[0], center[1], -10);
            quad.Scale(radius, radius, 1);
        }
    }

    Render(timeElapsed) {
        this._constants['resolution'] = vec4.fromValues(
            window.innerWidth, window.innerHeight, 0, 0);
        this._camera.UpdateConstants(this._constants);

        this._constants['gBuffer_Normal'] = null;
        this._constants['gBuffer_Position'] = null;
        this._constants['gBuffer_Colour'] = null;
        this._constants['gBuffer_Light'] = null;

        // делаем z-проход, записывая все только в z-буфер без шейдинга
        GL.bindFramebuffer(GL.FRAMEBUFFER, this._zFBO);
        GL.drawBuffers([GL.COLOR_ATTACHMENT0, GL.COLOR_ATTACHMENT1]);

        GL.clearColor(0.0, 0.0, 0.0, 0.0);
        GL.clearDepth(1.0);
        GL.enable(GL.DEPTH_TEST);
        GL.depthMask(true);
        GL.depthFunc(GL.LEQUAL);
        GL.clear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT);

        this._camera.UpdateConstants(this._constants);

        // фиксируем данные, получившиеся в результате z-прохода, для мешей
        for (let m of this._meshes) {
            m.Bind(this._constants, 'z');
            m.Draw();
        }

        GL.useProgram(null);
        GL.bindTexture(GL.TEXTURE_2D, null);

        // генерируем буфер освещения
        // берем нормаль, положение и рассчитываем вклад света, который будет учитываться при рендеринге
        GL.bindFramebuffer(GL.FRAMEBUFFER, this._lightFBO);
        GL.drawBuffers([GL.COLOR_ATTACHMENT0]);

        // если мы хотим добавить больше источников света, то просто
        // используем свойство блендинга (смешивания), чтобы они не мешали друг другу
        GL.clear(GL.COLOR_BUFFER_BIT);
        GL.disable(GL.DEPTH_TEST);
        GL.enable(GL.BLEND);
        GL.blendFunc(GL.ONE, GL.ONE);

        this._postCamera.UpdateConstants(this._constants);

        this._constants['gBuffer_Normal'] = this._normalTexture;
        this._constants['gBuffer_Position'] = this._positionTexture;

        // задаем освещение
        for (let l of this._lights) {
            l.UpdateConstants(this._constants);

            let quad = null;
            if (l.Type == 'Directional') {
                quad = this._quadDirectional;
            } else if (l.Type == 'Point') {
                quad = this._quadPoint;

                this._SetQuadSizeForLight(quad, l);
            }
            quad.Bind(this._constants, 'light');
            quad.Draw();
        }

        GL.useProgram(null);
        GL.bindTexture(GL.TEXTURE_2D, null);

        // генерируем буфер цвета
        GL.bindFramebuffer(GL.FRAMEBUFFER, this._colourFBO);
        GL.drawBuffers([GL.COLOR_ATTACHMENT0]);
        GL.disable(GL.BLEND);
        GL.depthMask(false);
        GL.enable(GL.DEPTH_TEST);

        this._camera.UpdateConstants(this._constants);

        this._constants['gBuffer_Colour'] = null;
        this._constants['gBuffer_Light'] = this._lightTexture;
        this._constants['gBuffer_Normal'] = null;
        this._constants['gBuffer_Position'] = null;

        // рисуем меши
        for (let m of this._meshes) {
            m.Bind(this._constants, 'colour');
            m.Draw();
        }

        GL.useProgram(null);
        GL.bindTexture(GL.TEXTURE_2D, null);
        GL.disable(GL.BLEND);

        // отрисовка на экран
        GL.bindFramebuffer(GL.FRAMEBUFFER, null);
        GL.disable(GL.DEPTH_TEST);
        GL.disable(GL.BLEND);

        this._postCamera.UpdateConstants(this._constants);

        this._constants['gQuadTexture'] = this._colourTexture;

        this._quadColour.Bind(this._constants, 'colour');
        this._quadColour.Draw();
    }
}

class LabDemo {
    constructor() {
        this._Initialize();
    }

    _Initialize() {
        this._renderer = new Renderer();

        window.addEventListener('resize', () => {
            this._OnWindowResize();
        }, false);

        this._Init();

        this._previousRAF = null;
        this._RAF();
    }

    _OnWindowResize() {
        this._renderer.Resize(window.innerWidth, window.innerHeight);
    }

    _Init() {
        this._CreateLights();
        this._CreateMeshes();
    }

    _CreateLights() {
        this._lights = [];
        /*
                let l1 = this._renderer.CreateLight('directional');
                const v1 = vec3.fromValues(Math.random(), Math.random(), Math.random());
                vec3.normalize(v1, v1);
        
                l1.SetColour(2, 2, 2);
                l1.SetDirection(50, 50, 50);
        
                this._lights.push({
                    light: l1,
                    type: 'dir'
                });
        */
        for (let i = -9; i <= 9; i++) {
            let l = this._renderer.CreateLight('point');

            const v = vec3.fromValues(Math.random(), Math.random(), Math.random());
            vec3.normalize(v, v);

            const p = vec3.fromValues(
                (Math.random() * 2 - 1) * 10,
                3,
                -Math.random() * 10 - 10);

            l.SetColour(v[0], v[1], v[2]);
            l.SetPosition(p[0], p[1], p[2]);
            l.SetRadius(4, 1);

            this._lights.push({
                light: l,
                type: 'point',
                position: p,
                acc: Math.random() * 10.0,
                accSpeed: Math.random() * 0.5 + 0.5,
            });
        }
    }

    _CreateMeshes() {
        this._meshes = [];

        let m = this._renderer.CreateMeshInstance(
            new Quad(),
            {
                shader: 'default',
                params: {
                    diffuseTexture: 'worn-bumpy-rock-albedo',
                    normalTexture: 'worn-bumpy-rock-normal',
                }
            });
        m.SetPosition(0, -2, -10);
        m.RotateX(-Math.PI * 0.5);
        m.Scale(50, 50, 1);

        for (let x = -5; x < 5; x++) {
            for (let y = 0; y < 10; y++) {
                let m;
                if (y == 5) {
                    m = this._renderer.CreateMeshInstance(
                        new CustomMesh(pyramidObj),
                        {
                            shader: 'default',
                            params: {
                                diffuseTexture: 'worn-bumpy-rock-albedo',
                                normalTexture: 'worn-bumpy-rock-normal',
                            }
                        });
                } else {
                    m = this._renderer.CreateMeshInstance(
                        new Cube(),
                        {
                            shader: 'default',
                            params: {
                                diffuseTexture: 'test-diffuse',
                                normalTexture: 'test-normal',
                            }
                        });
                }
                m.SetPosition(x * 4, 0, -y * 4);

                this._meshes.push(m);
            }
        }
    }

    _RAF() {
        requestAnimationFrame((t) => {
            if (this._previousRAF === null) {
                this._previousRAF = t;
            }

            this._RAF();
            this._Step(t - this._previousRAF);
            this._previousRAF = t;
        });
    }

    _Step(timeElapsed) {
        const timeElapsedS = timeElapsed * 0.001;

        for (let m of this._meshes) {
            m.RotateY(timeElapsedS);
        }

        for (let l of this._lights) {
            if (l.type != 'dir') {
                l.acc += timeElapsed * 0.001 * l.accSpeed;

                l.light.SetPosition(
                    l.position[0] + 10 * Math.cos(l.acc),
                    l.position[1],
                    l.position[2] + 10 * Math.sin(l.acc));
            }
        }

        /*
            let x = this._renderer._camera._position[0];
            let y = this._renderer._camera._position[1];
            let z = this._renderer._camera._position[2];
            this._renderer._camera.SetPosition(x * Math.cos(timeElapsedS) - z * Math.sin(timeElapsedS),
                y,
                z * Math.cos(timeElapsedS) + x * Math.sin(timeElapsedS));
        */
        this._renderer.Render(timeElapsedS);
    }
}


let _APP = null;

window.addEventListener('DOMContentLoaded', () => {
    _APP = new LabDemo();
});

window.addEventListener('keydown', function (event) {
    event.preventDefault();
    let x = _APP._renderer._camera._position[0];
    let y = _APP._renderer._camera._position[1];
    let z = _APP._renderer._camera._position[2];
    switch (event.keyCode) {
        case 37:                                      // влево
            _APP._renderer._camera.SetPosition(x - 1, y, z);
            break;
        case 38:                                      // вверх
            _APP._renderer._camera.SetPosition(x, y, z - 1);
            break;
        case 39:                                      // вправо
            _APP._renderer._camera.SetPosition(x + 1, y, z);
            break;
        case 40:                                      // вниз
            _APP._renderer._camera.SetPosition(x, y, z + 1);
            break;
        case 33:                                      // PgUp
            _APP._renderer._camera.SetPosition(x, y + 1, z);
            break;
        case 34:                                      // PgDn
            _APP._renderer._camera.SetPosition(x, y - 1, z);
            break;
    }
}); 