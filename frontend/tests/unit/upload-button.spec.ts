import { shallowMount, createLocalVue } from '@vue/test-utils'
import UploadButton from '@/components/UploadButton.vue';
import Vuetify from 'vuetify';

const localVue = createLocalVue()

describe('UploadButton.vue', () => {
  let vuetify

  beforeEach(() => {
    vuetify = new Vuetify()
  })

  const mountFunction = options => {
    return shallowMount(UploadButton, {
      localVue,
      vuetify,
      ...options,
    })
  }

  it('renders props.title when passed', () => {
    const title = 'upload a file';
    const wrapper = mountFunction({
      slots: {
        default: title,
      },
    });
    expect(wrapper.text()).toMatch(title);
  });
});
