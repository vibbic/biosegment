<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Edit Annotation</div>
      </v-card-title>
      <v-card-text>
        <AnnotationForm
          :annotation="annotationForm"
          title="Update Annotation"
        ></AnnotationForm>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit">Save</v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Annotation, AnnotationUpdate, AnnotationCreate } from '@/api';
import { defaultAnnotation } from '@/interfaces';
import AnnotationForm from '@/components/AnnotationForm.vue';
import {
  dispatchGetAnnotations,
  dispatchUpdateAnnotation,
} from '@/store/annotation/actions';
import { component } from 'vue/types/umd';
import { readOneAnnotation } from '@/store/annotation/getters';
import { filterUndefined, deepCopy } from '@/utils';

@Component({ components: { AnnotationForm } })
export default class EditAnnotation extends Vue {
  public annotationForm: AnnotationUpdate = deepCopy(this.annotation);
  public valid = false;

  public async mounted() {
    await dispatchGetAnnotations(this.$store);
    this.reset();
  }

  public reset() {
    this.annotationForm = deepCopy(this.annotation);
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredAnnotation: AnnotationUpdate = filterUndefined(this.annotationForm);
      await dispatchUpdateAnnotation(this.$store, {
        id: this.annotation.id,
        annotation: filteredAnnotation,
      });
      this.$router.push('/main/annotations');
    }
  }

  get annotation() {
    return readOneAnnotation(this.$store)(+this.$router.currentRoute.params.id);
  }
}
</script>
